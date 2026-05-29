// Copyright © FunAudioLLM contributors (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/FunAudioLLM/CosyVoice
// License: licenses/cosyvoice.txt

import Foundation
import MLX
import MLXNN

// MARK: - CosyVoice3 Streaming Constants

/// FSQ silent and breath tokens (from PyTorch CosyVoice3Model)
/// These are filtered during streaming to avoid excessive pauses
let cosyVoice3SilentTokens: Set<Int> = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
let cosyVoice3MaxSilentTokenNum = 5

// MARK: - CosyVoice3 Main Model

/// CosyVoice3 Text-to-Speech model
///
/// This model generates high-quality speech from text using:
/// 1. Qwen2-based LLM for speech token generation
/// 2. DiT-based Flow Matching for mel spectrogram synthesis
/// 3. Causal HiFi-GAN for waveform generation
///
/// Supports:
/// - Zero-shot voice matching with prompt audio
/// - Cross-lingual mode (no reference transcription needed)
/// - Instruct mode for style control
/// - Voice conversion
/// - Full streaming support
class CosyVoice3Model: Module {
  let config: CosyVoice3Config

  @ModuleInfo var llm: CosyVoice3LM?
  @ModuleInfo var flow: CosyVoice3FlowModule?
  @ModuleInfo var hifigan: CausalHiFTGenerator?

  var sampleRate: Int {
    config.hifigan.samplingRate
  }

  init(
    config: CosyVoice3Config = CosyVoice3Config(),
    llm: CosyVoice3LM? = nil,
    flow: CosyVoice3FlowModule? = nil,
    hifigan: CausalHiFTGenerator? = nil
  ) {
    self.config = config
    _llm.wrappedValue = llm
    _flow.wrappedValue = flow
    _hifigan.wrappedValue = hifigan
  }

  /// Generate speech tokens from text
  /// - Parameters:
  ///   - text: Input text token IDs (1, T)
  ///   - textLen: Text length (1,)
  ///   - promptText: Prompt text token IDs for voice matching (1, T_p)
  ///   - promptTextLen: Prompt text length (1,)
  ///   - promptSpeechToken: Prompt speech tokens for voice matching (1, T_s)
  ///   - promptSpeechTokenLen: Prompt speech token length (1,)
  ///   - embedding: Speaker embedding (unused, for API compat)
  ///   - sampling: Top-k sampling parameter
  ///   - maxTokenTextRatio: Maximum speech/text token ratio
  ///   - minTokenTextRatio: Minimum speech/text token ratio
  /// - Returns: Array of generated speech token IDs
  func generateTokens(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    embedding: MLXArray? = nil,
    sampling: Int = 25,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0
  ) throws -> [Int] {
    guard let llm else {
      throw CosyVoice3Error.modelNotLoaded
    }

    return try llm.inference(
      text: text,
      textLen: textLen,
      promptText: promptText,
      promptTextLen: promptTextLen,
      promptSpeechToken: promptSpeechToken,
      promptSpeechTokenLen: promptSpeechTokenLen,
      embedding: embedding,
      sampling: sampling,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )
  }

  /// Convert speech tokens to mel spectrogram
  /// - Parameters:
  ///   - tokens: Speech tokens (1, T)
  ///   - tokenLen: Token length (1,)
  ///   - promptToken: Prompt speech tokens (1, T_p)
  ///   - promptTokenLen: Prompt token length (1,)
  ///   - promptFeat: Prompt mel features (1, T_mel, D)
  ///   - promptFeatLen: Prompt feature length (1,)
  ///   - embedding: Speaker embedding (1, D_spk)
  ///   - finalize: Whether this is the final chunk
  ///   - nTimesteps: Number of diffusion steps
  ///   - streaming: Whether in streaming mode
  /// - Returns: Tuple of (mel_spectrogram, flow_cache)
  func tokensToMel(
    tokens: MLXArray,
    tokenLen: MLXArray,
    promptToken: MLXArray,
    promptTokenLen: MLXArray,
    promptFeat: MLXArray,
    promptFeatLen: MLXArray,
    embedding: MLXArray,
    finalize: Bool = true,
    nTimesteps: Int? = nil,
    streaming: Bool = false
  ) throws -> (MLXArray, MLXArray?) {
    guard let flow else {
      throw CosyVoice3Error.modelNotLoaded
    }

    return flow.inference(
      token: tokens,
      tokenLen: tokenLen,
      promptToken: promptToken,
      promptTokenLen: promptTokenLen,
      promptFeat: promptFeat,
      promptFeatLen: promptFeatLen,
      embedding: embedding,
      streaming: streaming,
      finalize: finalize,
      nTimesteps: nTimesteps ?? config.flow.nTimesteps
    )
  }

  /// Convert mel spectrogram to audio waveform
  /// - Parameters:
  ///   - mel: Mel spectrogram (1, D, T)
  ///   - finalize: Whether this is the final chunk
  /// - Returns: Audio waveform (1, T_audio)
  func melToAudio(mel: MLXArray, finalize: Bool = true) throws -> MLXArray {
    guard let hifigan else {
      throw CosyVoice3Error.modelNotLoaded
    }

    let (audio, _) = hifigan(mel, finalize: finalize)
    return audio
  }

  // MARK: - Speed control

  /// Resolves the effective speed factor for mel time-scaling.
  ///
  /// When `targetTokensPerTextToken` is provided, derives the factor automatically so the
  /// output's voiced tokens-per-text-token rate matches the target
  /// (`speed = generatedVoicedRatio / target`); otherwise uses `manualSpeed`. The result is
  /// clamped to `[0.7, 1.4]` to avoid extreme stretching. A factor `< 1` slows the output
  /// down.
  ///
  /// Silent/breath tokens are excluded from `generatedTokens` here, matching how
  /// `targetTokensPerTextToken` is computed from the reference (see
  /// `CosyVoice3Speaker.promptTokensPerTextTokenRatio`). Counting them on only one side
  /// would bias the speed: the generated text usually has fewer pauses than a reference, so
  /// an all-tokens generated ratio over a voiced-only target ratio comes out systematically
  /// too fast.
  static func resolveSpeed(
    manualSpeed: Float,
    targetTokensPerTextToken: Float?,
    generatedTokens: [Int],
    textLen: MLXArray
  ) -> Float {
    guard let target = targetTokensPerTextToken, target > 0 else {
      return manualSpeed
    }
    let textTokenCount = Float(textLen[0].item(Int32.self))
    guard textTokenCount > 0 else { return manualSpeed }
    let voicedCount = generatedTokens.reduce(0) { cosyVoice3SilentTokens.contains($1) ? $0 : $0 + 1 }
    let genRatio = Float(voicedCount) / textTokenCount
    return min(max(genRatio / target, 0.7), 1.4)
  }

  /// Linearly interpolates a mel spectrogram `[1, 80, T]` along the time axis (axis 2) to
  /// `T' = round(T / speed)`, matching `F.interpolate(mode: "linear", alignCorners: false)`
  /// with half-pixel sampling. Returns the input unchanged when `speed == 1` or `T < 2`.
  static func applySpeedToMel(_ mel: MLXArray, speed: Float) -> MLXArray {
    let t = mel.shape[2]
    guard speed != 1.0, t >= 2 else { return mel }
    let targetLen = max(1, Int((Float(t) / speed).rounded()))
    if targetLen == t { return mel }

    // align_corners == false, half-pixel: src(i) = (i + 0.5) * (T/T') - 0.5, clamped to [0, T-1]
    let scale = Float(t) / Float(targetLen)
    let dstIdx = MLXArray((0 ..< targetLen).map { Float($0) })
    var src = (dstIdx + 0.5) * scale - 0.5
    src = MLX.clip(src, min: MLXArray(Float(0)), max: MLXArray(Float(t - 1)))
    let lo = MLX.floor(src).asType(.int32)
    let hi = MLX.minimum(lo + Int32(1), MLXArray(Int32(t - 1)))
    let frac = (src - lo.asType(.float32)).reshaped(1, 1, targetLen)

    let melLo = mel.take(lo, axis: 2) // [1, 80, T']
    let melHi = mel.take(hi, axis: 2) // [1, 80, T']
    return melLo * (1 - frac) + melHi * frac
  }

  /// Full TTS pipeline: text -> audio (zero-shot mode)
  ///
  /// Mel-domain speed control mirrors the reference PyTorch implementation
  /// (`cosyvoice/cli/model.py` `token2wav`), which time-scales the flow-produced mel with
  /// `F.interpolate(tts_mel, size: Int(T / speed), mode: "linear")` before the vocoder.
  /// `speed < 1` adds mel frames -> longer, slower audio; frequency bins are untouched so
  /// pitch is preserved and HiFi-GAN resynthesizes naturally.
  /// - speed: manual factor (matches the reference `speed` argument).
  /// - targetTokensPerTextToken: when non-nil, the factor is derived from the actual
  ///   generated token count (overriding `speed`) so the output's tokens-per-text-token
  ///   rate tracks the target. Useful because zero-shot does not otherwise reproduce a
  ///   reference's speaking rate.
  func synthesize(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    sampling: Int = 25,
    nTimesteps: Int = 10,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0,
    speed: Float = 1.0,
    targetTokensPerTextToken: Float? = nil
  ) throws -> MLXArray {
    let totalStart = CFAbsoluteTimeGetCurrent()

    // Step 1: Generate speech tokens
    print("[LLM] Starting token generation (zero-shot)...")
    let llmStart = CFAbsoluteTimeGetCurrent()

    let tokens = try generateTokens(
      text: text,
      textLen: textLen,
      promptText: promptText,
      promptTextLen: promptTextLen,
      promptSpeechToken: promptSpeechToken,
      promptSpeechTokenLen: promptSpeechTokenLen,
      sampling: sampling,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )

    let llmTime = CFAbsoluteTimeGetCurrent() - llmStart
    print("[LLM] Generated \(tokens.count) tokens in \(String(format: "%.2f", llmTime))s (\(String(format: "%.1f", Double(tokens.count) / llmTime)) tokens/s)")

    if tokens.isEmpty {
      throw CosyVoice3Error.invalidInput("No tokens generated")
    }

    // Convert to array
    let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
    let tokenLen = MLXArray([Int32(tokens.count)])

    // Step 2: Convert tokens to mel spectrogram
    print("[Flow] Starting flow model...")
    let flowStart = CFAbsoluteTimeGetCurrent()

    let (mel, _) = try tokensToMel(
      tokens: tokenArray,
      tokenLen: tokenLen,
      promptToken: promptSpeechToken,
      promptTokenLen: promptSpeechTokenLen,
      promptFeat: promptMel,
      promptFeatLen: promptMelLen,
      embedding: speakerEmbedding,
      finalize: true,
      nTimesteps: nTimesteps
    )

    let flowTime = CFAbsoluteTimeGetCurrent() - flowStart
    print("[Flow] Completed in \(String(format: "%.2f", flowTime))s")

    // Time-scale the mel before the vocoder (see `synthesize` for details).
    let effectiveSpeed = Self.resolveSpeed(
      manualSpeed: speed,
      targetTokensPerTextToken: targetTokensPerTextToken,
      generatedTokens: tokens,
      textLen: textLen
    )
    let speedMel = Self.applySpeedToMel(mel, speed: effectiveSpeed)

    // Step 3: Convert mel to audio
    print("[HiFi-GAN] Converting mel to audio...")
    let hiStart = CFAbsoluteTimeGetCurrent()

    let audio = try melToAudio(mel: speedMel)

    let hiTime = CFAbsoluteTimeGetCurrent() - hiStart
    print("[HiFi-GAN] Completed in \(String(format: "%.2f", hiTime))s")

    let totalTime = CFAbsoluteTimeGetCurrent() - totalStart
    print("[TIMING SUMMARY] Total: \(String(format: "%.2f", totalTime))s | LLM: \(String(format: "%.2f", llmTime))s | Flow: \(String(format: "%.2f", flowTime))s | HiFi-GAN: \(String(format: "%.2f", hiTime))s")

    return audio
  }

  /// Zero-shot voice matching TTS pipeline
  ///
  /// This mode requires a transcription of the reference audio (promptText).
  func synthesizeZeroShot(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    sampling: Int = 25,
    nTimesteps: Int = 10,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0,
    speed: Float = 1.0,
    targetTokensPerTextToken: Float? = nil
  ) throws -> MLXArray {
    try synthesize(
      text: text,
      textLen: textLen,
      promptText: promptText,
      promptTextLen: promptTextLen,
      promptSpeechToken: promptSpeechToken,
      promptSpeechTokenLen: promptSpeechTokenLen,
      promptMel: promptMel,
      promptMelLen: promptMelLen,
      speakerEmbedding: speakerEmbedding,
      sampling: sampling,
      nTimesteps: nTimesteps,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio,
      speed: speed,
      targetTokensPerTextToken: targetTokensPerTextToken
    )
  }

  /// Cross-lingual TTS pipeline (no reference transcription needed)
  ///
  /// In this mode:
  /// - LLM receives NO promptText and NO promptSpeechToken
  /// - LLM generates speech tokens based purely on the input text
  /// - Flow model still uses promptSpeechToken and promptMel for speaker identity
  func synthesizeCrossLingual(
    text: MLXArray,
    textLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    sampling: Int = 25,
    nTimesteps: Int = 10,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0,
    speed: Float = 1.0,
    targetTokensPerTextToken: Float? = nil
  ) throws -> MLXArray {
    let totalStart = CFAbsoluteTimeGetCurrent()

    // Generate tokens WITHOUT prompt context
    let emptyPromptText = MLXArray.zeros([1, 0], dtype: .int32)
    let emptyPromptTextLen = MLXArray([Int32(0)])
    let emptySpeechToken = MLXArray.zeros([1, 0], dtype: .int32)
    let emptySpeechTokenLen = MLXArray([Int32(0)])

    print("[LLM] Starting token generation...")
    let llmStart = CFAbsoluteTimeGetCurrent()

    let tokens = try generateTokens(
      text: text,
      textLen: textLen,
      promptText: emptyPromptText,
      promptTextLen: emptyPromptTextLen,
      promptSpeechToken: emptySpeechToken,
      promptSpeechTokenLen: emptySpeechTokenLen,
      sampling: sampling,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )

    let llmTime = CFAbsoluteTimeGetCurrent() - llmStart
    print("[LLM] Generated \(tokens.count) tokens in \(String(format: "%.2f", llmTime))s (\(String(format: "%.1f", Double(tokens.count) / llmTime)) tokens/s)")

    if tokens.isEmpty {
      throw CosyVoice3Error.invalidInput("No tokens generated")
    }

    let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
    let tokenLen = MLXArray([Int32(tokens.count)])

    // Flow model uses prompt for speaker identity
    print("[Flow] Starting flow model...")
    let flowStart = CFAbsoluteTimeGetCurrent()

    let (mel, _) = try tokensToMel(
      tokens: tokenArray,
      tokenLen: tokenLen,
      promptToken: promptSpeechToken,
      promptTokenLen: promptSpeechTokenLen,
      promptFeat: promptMel,
      promptFeatLen: promptMelLen,
      embedding: speakerEmbedding,
      finalize: true,
      nTimesteps: nTimesteps
    )

    let flowTime = CFAbsoluteTimeGetCurrent() - flowStart
    print("[Flow] Completed in \(String(format: "%.2f", flowTime))s, mel shape: \(mel.shape)")

    // Time-scale the mel before the vocoder (see `synthesize` for details).
    let effectiveSpeed = Self.resolveSpeed(
      manualSpeed: speed,
      targetTokensPerTextToken: targetTokensPerTextToken,
      generatedTokens: tokens,
      textLen: textLen
    )
    let speedMel = Self.applySpeedToMel(mel, speed: effectiveSpeed)

    print("[HiFi-GAN] Converting mel to audio...")
    let hiStart = CFAbsoluteTimeGetCurrent()
    let audio = try melToAudio(mel: speedMel)
    let hiTime = CFAbsoluteTimeGetCurrent() - hiStart
    print("[HiFi-GAN] Completed in \(String(format: "%.2f", hiTime))s, audio shape: \(audio.shape)")

    let totalTime = CFAbsoluteTimeGetCurrent() - totalStart
    let audioSamples = audio.shape[0]
    let audioDuration = Double(audioSamples) / Double(sampleRate)
    let rtf = totalTime / audioDuration

    print("[TIMING SUMMARY] Total: \(String(format: "%.2f", totalTime))s | LLM: \(String(format: "%.2f", llmTime))s (\(String(format: "%.0f", llmTime / totalTime * 100))%) | Flow: \(String(format: "%.2f", flowTime))s (\(String(format: "%.0f", flowTime / totalTime * 100))%) | HiFi-GAN: \(String(format: "%.2f", hiTime))s (\(String(format: "%.0f", hiTime / totalTime * 100))%)")
    print("[TIMING SUMMARY] Audio: \(String(format: "%.2f", audioDuration))s | RTF: \(String(format: "%.2f", rtf))")

    return audio
  }

  /// Instruct-mode TTS pipeline with style control
  ///
  /// This allows controlling the style of speech generation with instructions
  /// like "Speak slowly and calmly" or "Read with excitement".
  func synthesizeInstruct(
    text: MLXArray,
    textLen: MLXArray,
    instructText: MLXArray,
    instructTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    sampling: Int = 25,
    nTimesteps: Int = 10,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0
  ) throws -> MLXArray {
    // Generate with instruct context
    let emptySpeechToken = MLXArray.zeros([1, 0], dtype: .int32)
    let emptySpeechTokenLen = MLXArray([Int32(0)])

    let tokens = try generateTokens(
      text: text,
      textLen: textLen,
      promptText: instructText,
      promptTextLen: instructTextLen,
      promptSpeechToken: emptySpeechToken,
      promptSpeechTokenLen: emptySpeechTokenLen,
      sampling: sampling,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )

    if tokens.isEmpty {
      throw CosyVoice3Error.invalidInput("No tokens generated")
    }

    let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)
    let tokenLen = MLXArray([Int32(tokens.count)])

    // Flow model uses prompt for speaker identity
    let (mel, _) = try tokensToMel(
      tokens: tokenArray,
      tokenLen: tokenLen,
      promptToken: promptSpeechToken,
      promptTokenLen: promptSpeechTokenLen,
      promptFeat: promptMel,
      promptFeatLen: promptMelLen,
      embedding: speakerEmbedding,
      finalize: true,
      nTimesteps: nTimesteps
    )

    return try melToAudio(mel: mel)
  }

  /// Voice Conversion (VC) pipeline: convert source speech to target speaker voice
  ///
  /// In this mode:
  /// - No LLM inference - source speech tokens are used directly
  /// - Flow model converts source tokens to target speaker voice
  func synthesizeVC(
    sourceSpeechToken: MLXArray,
    sourceSpeechTokenLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    nTimesteps: Int = 10
  ) throws -> MLXArray {
    // VC mode: use source speech tokens directly
    let (mel, _) = try tokensToMel(
      tokens: sourceSpeechToken,
      tokenLen: sourceSpeechTokenLen,
      promptToken: promptSpeechToken,
      promptTokenLen: promptSpeechTokenLen,
      promptFeat: promptMel,
      promptFeatLen: promptMelLen,
      embedding: speakerEmbedding,
      finalize: true,
      nTimesteps: nTimesteps
    )

    return try melToAudio(mel: mel)
  }

  /// Streaming speech synthesis with chunked output
  ///
  /// Generates audio in chunks as tokens are produced, reducing latency.
  /// This is the true streaming implementation that yields audio as tokens are generated.
  ///
  /// - Parameters:
  ///   - text: Input text token IDs (1, T)
  ///   - textLen: Text length (1,)
  ///   - promptText: Prompt text token IDs (1, T_p)
  ///   - promptTextLen: Prompt text length (1,)
  ///   - promptSpeechToken: Prompt speech tokens (1, T_s)
  ///   - promptSpeechTokenLen: Prompt speech token length (1,)
  ///   - promptMel: Prompt mel spectrogram (1, T_mel, D)
  ///   - promptMelLen: Prompt mel length (1,)
  ///   - speakerEmbedding: Speaker embedding (1, D_spk)
  ///   - sampling: Top-k sampling parameter
  ///   - nTimesteps: Number of flow matching steps
  ///   - chunkSize: Number of tokens per audio chunk (default: 25, must match training)
  ///   - filterSilentTokens: Whether to filter consecutive silent tokens
  ///   - maxTokenTextRatio: Maximum speech/text ratio
  ///   - minTokenTextRatio: Minimum speech/text ratio
  /// - Returns: AsyncThrowingStream of audio chunks as MLXArray
  func synthesizeStreaming(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    llmPromptSpeechToken: MLXArray? = nil,
    llmPromptSpeechTokenLen: MLXArray? = nil,
    sampling: Int = 25,
    nTimesteps: Int = 10,
    chunkSize: Int = 25,
    filterSilentTokens: Bool = true,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0
  ) -> AsyncThrowingStream<MLXArray, Error> {
    guard let llm, let flow, let hifigan else {
      return AsyncThrowingStream { continuation in
        continuation.finish(throwing: CosyVoice3Error.modelNotLoaded)
      }
    }

    // Get the token stream from LLM
    let tokenStream = llm.inferenceStreamAsync(
      text: text,
      textLen: textLen,
      promptText: promptText,
      promptTextLen: promptTextLen,
      promptSpeechToken: llmPromptSpeechToken ?? promptSpeechToken,
      promptSpeechTokenLen: llmPromptSpeechTokenLen ?? promptSpeechTokenLen,
      sampling: sampling,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )

    // Calculate prompt token padding to align first chunk to chunk_size boundary
    // This matches PyTorch: prompt_token_pad = ceil(prompt_len / chunk_size) * chunk_size - prompt_len
    let promptLen = Int(promptSpeechTokenLen[0].item(Int32.self))
    let promptTokenPad = Int((Float(promptLen) / Float(chunkSize)).rounded(.up)) * chunkSize - promptLen

    // Use pull-based pattern with a state-encapsulating iterator box.
    // Marked @unchecked Sendable because AsyncThrowingStream guarantees sequential
    // access to the unfolding closure - next() is never called concurrently.
    final class StreamingIteratorBox: @unchecked Sendable {
      var tokenIterator: AsyncThrowingStream<Int, Error>.Iterator
      var speechTokens: [Int] = []
      var tokenOffset = 0
      var melCache: MLXArray?
      var speechOffset = 0
      var curSilentTokenNum = 0
      var finished = false
      var currentChunkSize: Int

      let flow: CosyVoice3FlowModule
      let hifigan: CausalHiFTGenerator
      let promptSpeechToken: MLXArray
      let promptSpeechTokenLen: MLXArray
      let promptMel: MLXArray
      let promptMelLen: MLXArray
      let speakerEmbedding: MLXArray
      let chunkSize: Int
      let tokenMaxHopLen: Int
      let promptTokenPad: Int
      let filterSilentTokens: Bool
      let nTimesteps: Int
      let preLookaheadLen: Int
      let tokenMelRatio: Int

      init(
        tokenStream: AsyncThrowingStream<Int, Error>,
        flow: CosyVoice3FlowModule,
        hifigan: CausalHiFTGenerator,
        promptSpeechToken: MLXArray,
        promptSpeechTokenLen: MLXArray,
        promptMel: MLXArray,
        promptMelLen: MLXArray,
        speakerEmbedding: MLXArray,
        chunkSize: Int,
        promptTokenPad: Int,
        filterSilentTokens: Bool,
        nTimesteps: Int
      ) {
        tokenIterator = tokenStream.makeAsyncIterator()
        self.flow = flow
        self.hifigan = hifigan
        self.promptSpeechToken = promptSpeechToken
        self.promptSpeechTokenLen = promptSpeechTokenLen
        self.promptMel = promptMel
        self.promptMelLen = promptMelLen
        self.speakerEmbedding = speakerEmbedding
        self.chunkSize = chunkSize
        tokenMaxHopLen = chunkSize * 4
        self.promptTokenPad = promptTokenPad
        self.filterSilentTokens = filterSilentTokens
        self.nTimesteps = nTimesteps
        preLookaheadLen = flow.preLookaheadLen
        tokenMelRatio = flow.tokenMelRatio
        currentChunkSize = chunkSize
      }

      func next() async throws -> MLXArray? {
        // If we've already finished, return nil
        if finished { return nil }

        // Match the original PyTorch implementation's streaming behavior: grow
        // the token hop size after each
        // emitted chunk to reduce duplicate work.
        let thisChunkSize = tokenOffset == 0 ? currentChunkSize + promptTokenPad : currentChunkSize

        // Keep consuming tokens until we have enough for a chunk
        // Matches PyTorch: len(tokens) - token_offset >= this_chunk_size + pre_lookahead_len
        while speechTokens.count - tokenOffset < thisChunkSize + preLookaheadLen {
          guard let token = try await tokenIterator.next() else {
            // Token stream ended - process final chunk if we have remaining tokens
            if speechTokens.count > tokenOffset {
              let result = try processChunk(finalize: true)
              finished = true
              return result
            }
            finished = true
            return nil
          }

          // Filter consecutive silent tokens (matches PyTorch llm_job)
          if filterSilentTokens, cosyVoice3SilentTokens.contains(token) {
            curSilentTokenNum += 1
            if curSilentTokenNum > cosyVoice3MaxSilentTokenNum {
              continue
            }
          } else {
            curSilentTokenNum = 0
          }

          speechTokens.append(token)
        }

        // We have enough tokens for an intermediate chunk
        let result = try processChunk(finalize: false)
        tokenOffset += thisChunkSize
        currentChunkSize = min(tokenMaxHopLen, currentChunkSize * 2)
        return result
      }

      private func processChunk(finalize: Bool) throws -> MLXArray? {
        // Take all tokens up to current position + lookahead
        let thisChunkSize = tokenOffset == 0 ? currentChunkSize + promptTokenPad : currentChunkSize
        let endIdx = finalize ? speechTokens.count : tokenOffset + thisChunkSize + preLookaheadLen
        let chunkTokens = Array(speechTokens[0 ..< endIdx])
        let tokenArray = MLXArray(chunkTokens.map { Int32($0) }).reshaped(1, -1)
        let tokenLen = MLXArray([Int32(chunkTokens.count)])

        // Generate mel via flow
        // Note: PyTorch uses streaming=False for final chunk (full attention)
        let (mel, _) = flow.inference(
          token: tokenArray,
          tokenLen: tokenLen,
          promptToken: promptSpeechToken,
          promptTokenLen: promptSpeechTokenLen,
          promptFeat: promptMel,
          promptFeatLen: promptMelLen,
          embedding: speakerEmbedding,
          streaming: !finalize,
          finalize: finalize,
          nTimesteps: nTimesteps
        )

        // Slice mel to get only new portion (from token_offset)
        let melNew = mel[0..., 0..., (tokenOffset * tokenMelRatio)...]

        // Accumulate mel (matches PyTorch caching strategy)
        if let cache = melCache {
          melCache = MLX.concatenated([cache, melNew], axis: 2)
        } else {
          melCache = melNew
        }

        // Generate audio from accumulated mel
        let (audio, _) = hifigan(melCache!, finalize: finalize)

        // Extract only new audio (from speech_offset)
        if audio.shape[audio.ndim - 1] > speechOffset {
          let chunkAudio = audio[0..., speechOffset...].squeezed(axis: 0)
          speechOffset += chunkAudio.shape[0]
          eval(chunkAudio)
          return chunkAudio
        }

        return nil
      }
    }

    let box = StreamingIteratorBox(
      tokenStream: tokenStream,
      flow: flow,
      hifigan: hifigan,
      promptSpeechToken: promptSpeechToken,
      promptSpeechTokenLen: promptSpeechTokenLen,
      promptMel: promptMel,
      promptMelLen: promptMelLen,
      speakerEmbedding: speakerEmbedding,
      chunkSize: chunkSize,
      promptTokenPad: promptTokenPad,
      filterSilentTokens: filterSilentTokens,
      nTimesteps: nTimesteps
    )

    return AsyncThrowingStream { try await box.next() }
  }

  /// Synchronous streaming synthesis (for backwards compatibility)
  ///
  /// Collects all chunks and returns them as an array.
  func synthesizeStreamingSync(
    text: MLXArray,
    textLen: MLXArray,
    promptText: MLXArray,
    promptTextLen: MLXArray,
    promptSpeechToken: MLXArray,
    promptSpeechTokenLen: MLXArray,
    promptMel: MLXArray,
    promptMelLen: MLXArray,
    speakerEmbedding: MLXArray,
    llmPromptSpeechToken: MLXArray? = nil,
    llmPromptSpeechTokenLen: MLXArray? = nil,
    sampling: Int = 25,
    nTimesteps: Int = 10,
    chunkSize: Int = 25,
    filterSilentTokens: Bool = true,
    maxTokenTextRatio: Float = 20.0,
    minTokenTextRatio: Float = 2.0
  ) async throws -> [MLXArray] {
    var chunks: [MLXArray] = []

    let stream = synthesizeStreaming(
      text: text,
      textLen: textLen,
      promptText: promptText,
      promptTextLen: promptTextLen,
      promptSpeechToken: promptSpeechToken,
      promptSpeechTokenLen: promptSpeechTokenLen,
      promptMel: promptMel,
      promptMelLen: promptMelLen,
      speakerEmbedding: speakerEmbedding,
      llmPromptSpeechToken: llmPromptSpeechToken,
      llmPromptSpeechTokenLen: llmPromptSpeechTokenLen,
      sampling: sampling,
      nTimesteps: nTimesteps,
      chunkSize: chunkSize,
      filterSilentTokens: filterSilentTokens,
      maxTokenTextRatio: maxTokenTextRatio,
      minTokenTextRatio: minTokenTextRatio
    )

    for try await chunk in stream {
      chunks.append(chunk)
    }

    return chunks
  }
}

// MARK: - Conditionals Container

/// Container for precomputed reference audio conditionals
struct CosyVoice3Conditionals: @unchecked Sendable {
  let promptSpeechToken: MLXArray
  let promptSpeechTokenLen: MLXArray
  let promptMel: MLXArray
  let promptMelLen: MLXArray
  let speakerEmbedding: MLXArray
  let promptText: MLXArray?
  let promptTextLen: MLXArray?
  /// Token count of the prompt text excluding the system prefix
  /// ("You are a helpful assistant.<|endofprompt|>"). Used to compute the reference's true
  /// speech-token/text-token ratio (the auto speed target) without the prefix inflating the
  /// denominator. `nil` when no reference text was supplied (cross-lingual).
  var promptTextLenWithoutPrefix: Int? = nil
}

// MARK: - CosyVoice3FlowModule

/// Flow module that wraps PreLookaheadLayer + DiT CFM for CosyVoice3
class CosyVoice3FlowModule: Module {
  let inputSize: Int
  let outputSize: Int
  let spkEmbedDim: Int
  let vocabSize: Int
  let inputFrameRate: Int
  let tokenMelRatio: Int
  let preLookaheadLen: Int
  let nTimesteps: Int

  @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
  @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
  @ModuleInfo(key: "pre_lookahead_layer") var preLookaheadLayer: CosyVoice3PreLookaheadLayer
  @ModuleInfo var decoder: CosyVoice3ConditionalCFM

  init(
    inputSize: Int = 80,
    outputSize: Int = 80,
    spkEmbedDim: Int = 192,
    vocabSize: Int = 6561,
    inputFrameRate: Int = 25,
    tokenMelRatio: Int = 2,
    preLookaheadLen: Int = 3,
    nTimesteps: Int = 10,
    preLookahead: CosyVoice3PreLookaheadLayer? = nil,
    decoder: CosyVoice3ConditionalCFM? = nil
  ) {
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.spkEmbedDim = spkEmbedDim
    self.vocabSize = vocabSize
    self.inputFrameRate = inputFrameRate
    self.tokenMelRatio = tokenMelRatio
    self.preLookaheadLen = preLookaheadLen
    self.nTimesteps = nTimesteps

    _inputEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: inputSize)
    _spkEmbedAffineLayer.wrappedValue = Linear(spkEmbedDim, outputSize)

    if let layer = preLookahead {
      _preLookaheadLayer.wrappedValue = layer
    } else {
      _preLookaheadLayer.wrappedValue = CosyVoice3PreLookaheadLayer(
        inChannels: inputSize,
        channels: 1024,
        preLookaheadLen: preLookaheadLen
      )
    }

    if let dec = decoder {
      _decoder.wrappedValue = dec
    } else {
      // Create a default CFM with nil estimator - will need to be set before use
      _decoder.wrappedValue = CosyVoice3ConditionalCFM(estimator: nil)
    }
  }

  /// Inference: convert tokens to mel spectrogram
  func inference(
    token: MLXArray,
    tokenLen: MLXArray,
    promptToken: MLXArray,
    promptTokenLen: MLXArray,
    promptFeat: MLXArray,
    promptFeatLen _: MLXArray,
    embedding: MLXArray,
    streaming: Bool = false,
    finalize: Bool = true,
    nTimesteps: Int? = nil
  ) -> (MLXArray, MLXArray?) {
    // Normalize speaker embedding
    let norm = MLX.sqrt(MLX.sum(embedding * embedding, axis: 1, keepDims: true))
    let embeddingNorm = embedding / (norm + 1e-8)
    let spks = spkEmbedAffineLayer(embeddingNorm)

    // Concatenate prompt and target tokens
    let fullToken = MLX.concatenated([promptToken, token], axis: 1)
    let fullTokenLen = tokenLen + promptTokenLen

    // Create mask for token embedding
    let maxTokenLen = fullToken.shape[1]
    let seqRange = MLXArray(0 ..< Int32(maxTokenLen))
    let tokenMask = (seqRange .< fullTokenLen.expandedDimensions(axis: 1)).asType(.float32)
    let tokenMaskExpanded = tokenMask.expandedDimensions(axis: -1) // (B, T, 1)

    // Clip tokens to valid embedding range
    let numEmbeddings = inputEmbedding.weight.shape[0]
    let clippedToken = MLX.clip(fullToken, min: 0, max: numEmbeddings - 1)

    // Embed tokens and apply mask
    var tokenEmb = inputEmbedding(clippedToken)
    tokenEmb = tokenEmb * tokenMaskExpanded

    // Process through PreLookaheadLayer
    // In streaming mode (finalize=false), split tokens into main + lookahead context
    // so the convolution sees real future tokens instead of zero padding
    var encoderOut: MLXArray
    if finalize {
      // Non-streaming: no context needed (layer pads with zeros)
      encoderOut = preLookaheadLayer(tokenEmb, context: nil)
    } else {
      // Streaming: split into main tokens and lookahead context
      let seqLen = tokenEmb.shape[1]
      let mainTokens = tokenEmb[0..., 0 ..< (seqLen - preLookaheadLen), 0...]
      let context = tokenEmb[0..., (seqLen - preLookaheadLen)..., 0...]
      encoderOut = preLookaheadLayer(mainTokens, context: context)
    }

    // Upsample by tokenMelRatio
    encoderOut = MLX.repeated(encoderOut, count: tokenMelRatio, axis: 1)

    // Calculate mel lengths
    let melLen1 = promptFeat.shape[1]
    let melLen2 = encoderOut.shape[1] - melLen1

    // Prepare conditioning
    let totalLen = melLen1 + melLen2
    var conds: MLXArray
    if melLen1 > 0 {
      let promptSlice = promptFeat[0..., 0 ..< melLen1, 0...]
      let zerosPad = MLXArray.zeros([1, melLen2, outputSize])
      conds = MLX.concatenated([promptSlice, zerosPad], axis: 1)
    } else {
      conds = MLXArray.zeros([1, totalLen, outputSize])
    }
    conds = conds.swappedAxes(1, 2) // (1, D, T)

    // Create mask for decoder
    let mask = MLXArray.ones([1, 1, totalLen])

    // Transpose encoder output to (B, C, T) for decoder - mu stays at input_size dims
    let mu = encoderOut.swappedAxes(1, 2)

    // Run CFM decoder
    let (mel, cache) = decoder(
      mu: mu,
      mask: mask,
      spks: spks,
      cond: conds,
      nTimesteps: nTimesteps ?? self.nTimesteps,
      streaming: streaming
    )

    // Extract only the new portion (after prompt)
    let outputMel = mel[0..., 0..., melLen1...]

    return (outputMel, cache)
  }
}
