import { config as loadEnv } from "dotenv";
import { z } from "zod";
import { logger } from "../utils/logger";

loadEnv();

const envSchema = z.object({
  NODE_ENV: z.string().default("development"),
  PORT: z.string().default("4000"),
  MONGODB_URI: z.string().url(),
  // Embedding options - using local transformers.js
  EMBEDDING_MODEL: z
    .string()
    .default("sentence-transformers/all-MiniLM-L6-v2"), // Local model using transformers.js (384 dims)
  // Vector DB options (use one)
  VECTOR_DB_PROVIDER: z
    .enum(["qdrant", "pinecone", "chroma"])
    .default("qdrant"),
  VECTOR_DB_URL: z.string().url().optional(), // For Qdrant/Chroma
  VECTOR_DB_API_KEY: z.string().optional(), // For Qdrant Cloud/Pinecone
  PINECONE_API_KEY: z.string().optional(), // For Pinecone
  PINECONE_INDEX: z.string().optional(), // For Pinecone
  PINECONE_ENVIRONMENT: z.string().optional(), // For Pinecone (legacy)
  VECTOR_COLLECTION: z.string().default("movies"),
  VECTOR_DIMENSION: z.coerce.number().default(384), // HF all-MiniLM-L6-v2 uses 384
});

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  logger.error(
    "Invalid environment configuration",
    parsed.error.flatten().fieldErrors
  );
  throw new Error("Invalid environment configuration");
}

export const env = {
  ...parsed.data,
  port: Number(parsed.data.PORT),
};
