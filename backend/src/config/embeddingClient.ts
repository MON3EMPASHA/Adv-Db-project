import OpenAI from 'openai';
import { env } from './env';

let openaiClient: OpenAI | null = null;

export const getEmbeddingClient = (): OpenAI => {
  if (!env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not configured');
  }

  if (!openaiClient) {
    openaiClient = new OpenAI({ apiKey: env.OPENAI_API_KEY });
  }

  return openaiClient;
};





