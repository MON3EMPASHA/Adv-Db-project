import { env } from "../config/env";
import { logger } from "../utils/logger";

// Cache the pipeline instance to avoid re-initializing on every call
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let localPipeline: any = null;
let pipelineModel: string | null = null;
// Cache the pipeline import to avoid re-importing the module
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pipelineModule: any = null;

/**
 * Initialize the local embedding pipeline using transformers.js
 * This runs the model locally in Node.js without any API calls
 * Uses dynamic import() because @xenova/transformers is an ES module
 */
const initializeLocalPipeline = async (): Promise<any> => {
  const modelName = env.EMBEDDING_MODEL;

  // If pipeline already initialized with the same model, reuse it
  if (localPipeline && pipelineModel === modelName) {
    return localPipeline;
  }

  try {
    logger.info(
      `Initializing local embedding pipeline with model: ${modelName}`
    );
    logger.info(
      "Note: First run will download the model (may take a few minutes)"
    );

    // Dynamic import of @xenova/transformers (required because it's an ES module)
    if (!pipelineModule) {
      pipelineModule = await import("@xenova/transformers");
    }

    const { pipeline } = pipelineModule;

    // Initialize the feature-extraction pipeline
    // This downloads and loads the model on first use
    // Try quantized first, fallback to non-quantized if not available
    try {
      localPipeline = await pipeline("feature-extraction", modelName, {
        quantized: true, // Use quantized model for faster loading and smaller size
      });
    } catch (quantizedError) {
      logger.info(
        `Quantized model not available, trying non-quantized version...`
      );
      localPipeline = await pipeline("feature-extraction", modelName, {
        quantized: false, // Fallback to non-quantized
      });
    }

    pipelineModel = modelName;
    logger.info(`✓ Local embedding pipeline initialized successfully`);
    return localPipeline;
  } catch (error) {
    logger.error(`Failed to initialize local embedding pipeline:`, error);
    throw new Error(
      `Failed to initialize local embedding model: ${
        error instanceof Error ? error.message : "Unknown error"
      }`
    );
  }
};

/**
 * Apply mean pooling to token embeddings to get sentence-level embedding
 * Token embeddings are [num_tokens, embedding_dim], we want [embedding_dim]
 */
const applyMeanPooling = (tokenEmbeddings: number[][]): number[] => {
  if (tokenEmbeddings.length === 0 || !tokenEmbeddings[0]) {
    throw new Error("Cannot pool empty token embeddings");
  }

  const embeddingDim = tokenEmbeddings[0]!.length;
  const pooled = new Array(embeddingDim).fill(0);

  // Sum all token embeddings
  for (const tokenEmbedding of tokenEmbeddings) {
    for (let i = 0; i < embeddingDim; i++) {
      pooled[i] += tokenEmbedding[i];
    }
  }

  // Average
  for (let i = 0; i < embeddingDim; i++) {
    pooled[i] /= tokenEmbeddings.length;
  }

  return pooled;
};

/**
 * Generate embeddings using local transformers.js model
 * Runs entirely in Node.js, no API calls needed
 *
 * Based on @xenova/transformers API:
 *   const extractor = await pipeline("feature-extraction", model);
 *   const result = await extractor("text");
 *   console.log(result[0]); // embedding vector
 */
const generateLocalEmbedding = async (text: string): Promise<number[]> => {
  try {
    const extractor = await initializeLocalPipeline();

    // Generate embedding - result[0] contains the embedding(s)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const result: any = await extractor(text);

    // Extract embedding from result
    // Transformers.js returns token-level embeddings that need pooling
    let tokenEmbeddings: number[][];

    if (Array.isArray(result)) {
      if (result.length === 0) {
        throw new Error("Empty result array from embedding pipeline");
      }

      // result[0] contains the embeddings - could be 2D [tokens, dims] or 1D [dims]
      const firstResult = result[0];

      if (Array.isArray(firstResult)) {
        // Check if it's 2D array [tokens, dims] or 1D array [dims]
        if (firstResult.length > 0 && Array.isArray(firstResult[0])) {
          // 2D: [[dim1, dim2, ...], [dim1, dim2, ...], ...] - token embeddings
          tokenEmbeddings = firstResult as number[][];
        } else {
          // 1D: [dim1, dim2, ...] - might already be pooled, or single token
          // If dimension matches expected (384), assume it's already sentence-level
          const embeddingDim = (firstResult as number[]).length;
          if (embeddingDim === env.VECTOR_DIMENSION) {
            return firstResult as number[];
          }
          // Otherwise, treat as single token embedding (unlikely but handle it)
          tokenEmbeddings = [firstResult as number[]];
        }
      } else {
        throw new Error(
          "Unexpected result structure - first element is not an array"
        );
      }
    } else if (result && typeof result === "object") {
      // Handle tensor-like object - try common properties
      if ("data" in result) {
        const data = (
          result as {
            data: number[] | Float32Array | number[][];
          }
        ).data;
        if (Array.isArray(data)) {
          if (Array.isArray(data[0])) {
            tokenEmbeddings = data as number[][];
          } else {
            const embeddingDim = (data as number[]).length;
            if (embeddingDim === env.VECTOR_DIMENSION) {
              return data as number[];
            }
            tokenEmbeddings = [data as number[]];
          }
        } else {
          // Single Float32Array - convert to 2D if needed
          const arr = Array.from(data as Float32Array);
          const embeddingDim = env.VECTOR_DIMENSION;
          if (arr.length === embeddingDim) {
            return arr;
          }
          // Reshape: assume it's [tokens * dims], reshape to [tokens, dims]
          const numTokens = Math.floor(arr.length / embeddingDim);
          tokenEmbeddings = [];
          for (let i = 0; i < numTokens; i++) {
            tokenEmbeddings.push(
              arr.slice(i * embeddingDim, (i + 1) * embeddingDim)
            );
          }
        }
      } else if ("tolist" in result && typeof result.tolist === "function") {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const listResult = (result as any).tolist();
        if (Array.isArray(listResult)) {
          if (Array.isArray(listResult[0])) {
            tokenEmbeddings = listResult as number[][];
          } else {
            const embeddingDim = (listResult as number[]).length;
            if (embeddingDim === env.VECTOR_DIMENSION) {
              return listResult as number[];
            }
            tokenEmbeddings = [listResult as number[]];
          }
        } else {
          throw new Error("tolist() did not return an array");
        }
      } else {
        throw new Error(`Unexpected result format: ${typeof result}`);
      }
    } else {
      throw new Error(`Unexpected result type: ${typeof result}`);
    }

    // Apply mean pooling to get sentence-level embedding
    const embedding = applyMeanPooling(tokenEmbeddings);

    if (!embedding || embedding.length === 0) {
      throw new Error("Invalid embedding result from local model");
    }

    return embedding;
  } catch (error) {
    logger.error("Failed to generate local embedding:", error);
    throw new Error(
      `Local embedding generation failed: ${
        error instanceof Error ? error.message : "Unknown error"
      }`
    );
  }
};

/**
 * Generate embeddings for text using local transformers.js
 */
export const generateEmbedding = async (text: string): Promise<number[]> => {
  if (!text || !text.trim()) {
    throw new Error("Cannot generate embedding for empty text");
  }

  try {
    return await generateLocalEmbedding(text);
  } catch (error) {
    logger.error("Failed to generate embedding", error);
    throw error;
  }
};
