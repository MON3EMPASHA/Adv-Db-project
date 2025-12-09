import { generateEmbedding } from '../services/embeddingService';
import { env } from '../config/env';
import { logger } from '../utils/logger';

/**
 * Test embedding generation with various text samples
 */
const testEmbedding = async () => {
  try {
    logger.info('Starting embedding model test...');
    logger.info(`Embedding Provider: ${env.EMBEDDING_PROVIDER}`);
    logger.info(`Embedding Model: ${env.EMBEDDING_MODEL}`);
    logger.info(`Expected Dimension: ${env.VECTOR_DIMENSION}`);

    // Test samples with different characteristics
    const testSamples = [
      {
        name: 'Short text',
        text: 'The Matrix is a science fiction action film.',
      },
      {
        name: 'Medium text',
        text: 'The Matrix is a 1999 science fiction action film written and directed by the Wachowskis. It stars Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, and Hugo Weaving.',
      },
      {
        name: 'Long text',
        text: 'The Matrix is a 1999 science fiction action film written and directed by the Wachowskis. It stars Keanu Reeves as Neo, a computer programmer who discovers that reality as he knows it is a simulation called the Matrix. Laurence Fishburne plays Morpheus, who offers Neo the choice between a red pill and a blue pill. Carrie-Anne Moss portrays Trinity, a hacker who helps Neo escape. The film explores themes of reality, identity, and control in a dystopian future where machines have enslaved humanity.',
      },
      {
        name: 'Movie description',
        text: 'A mind-bending journey into a dystopian future where humanity is trapped in a simulated reality. Follow Neo as he uncovers the truth and fights to free mankind from machine control.',
      },
      {
        name: 'Genre-focused text',
        text: 'Science fiction action thriller with cyberpunk elements, exploring artificial intelligence and virtual reality themes.',
      },
    ];

    logger.info(`\nTesting ${testSamples.length} text samples...\n`);

    const results: Array<{
      name: string;
      textLength: number;
      embedding: number[];
      dimension: number;
      duration: number;
      sampleValues: number[];
    }> = [];

    for (const sample of testSamples) {
      logger.info(`Testing: ${sample.name}`);
      logger.info(`  Text: "${sample.text.substring(0, 60)}..."`);
      logger.info(`  Text length: ${sample.text.length} characters`);

      const startTime = Date.now();
      try {
        const embedding = await generateEmbedding(sample.text);
        const duration = Date.now() - startTime;

        // Validate embedding
        if (!Array.isArray(embedding)) {
          throw new Error('Embedding is not an array');
        }

        if (embedding.length === 0) {
          throw new Error('Embedding is empty');
        }

        if (embedding.length !== env.VECTOR_DIMENSION) {
          logger.warn(
            `  ⚠ Dimension mismatch: Expected ${env.VECTOR_DIMENSION}, got ${embedding.length}`,
          );
        }

        // Check for valid numeric values
        const hasInvalidValues = embedding.some((val) => !Number.isFinite(val));
        if (hasInvalidValues) {
          throw new Error('Embedding contains invalid (non-finite) values');
        }

        // Calculate some statistics
        const min = Math.min(...embedding);
        const max = Math.max(...embedding);
        const avg = embedding.reduce((a, b) => a + b, 0) / embedding.length;
        const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));

        results.push({
          name: sample.name,
          textLength: sample.text.length,
          embedding,
          dimension: embedding.length,
          duration,
          sampleValues: embedding.slice(0, 5),
        });

        logger.info(`  ✓ Generated embedding in ${duration}ms`);
        logger.info(`  ✓ Dimension: ${embedding.length}`);
        logger.info(`  ✓ Value range: [${min.toFixed(4)}, ${max.toFixed(4)}]`);
        logger.info(`  ✓ Average value: ${avg.toFixed(4)}`);
        logger.info(`  ✓ Vector magnitude: ${magnitude.toFixed(4)}`);
        logger.info(`  ✓ Sample values: [${embedding.slice(0, 5).map((v) => v.toFixed(4)).join(', ')}]`);
        logger.info('');
      } catch (error) {
        const duration = Date.now() - startTime;
        logger.error(`  ❌ Failed after ${duration}ms:`, error);
        throw error;
      }
    }

    // Test embedding similarity
    logger.info('\nTesting embedding similarity...');
    if (results.length >= 2) {
      const embedding1 = results[0].embedding;
      const embedding2 = results[1].embedding;

      // Cosine similarity
      const dotProduct = embedding1.reduce((sum, val, idx) => sum + val * embedding2[idx], 0);
      const magnitude1 = Math.sqrt(embedding1.reduce((sum, val) => sum + val * val, 0));
      const magnitude2 = Math.sqrt(embedding2.reduce((sum, val) => sum + val * val, 0));
      const cosineSimilarity = dotProduct / (magnitude1 * magnitude2);

      logger.info(`  Cosine similarity between "${results[0].name}" and "${results[1].name}": ${cosineSimilarity.toFixed(4)}`);
      logger.info(`  (Values close to 1 indicate similar semantic meaning)`);
    }

    // Summary
    logger.info('\n✅ Embedding model test completed successfully!');
    logger.info('\nSummary:');
    logger.info(`  Model: ${env.EMBEDDING_MODEL}`);
    logger.info(`  Provider: ${env.EMBEDDING_PROVIDER}`);
    logger.info(`  Expected dimension: ${env.VECTOR_DIMENSION}`);
    logger.info(`  Actual dimension: ${results[0]?.dimension || 'N/A'}`);
    logger.info(`  Samples tested: ${results.length}`);
    logger.info(`  Average generation time: ${(results.reduce((sum, r) => sum + r.duration, 0) / results.length).toFixed(0)}ms`);

    if (results.every((r) => r.dimension === env.VECTOR_DIMENSION)) {
      logger.info(`  ✓ All embeddings have correct dimension`);
    } else {
      logger.warn(`  ⚠ Some embeddings have incorrect dimensions`);
    }

    logger.info('\nModel Information:');
    logger.info(`  - This model can convert text into ${results[0]?.dimension || env.VECTOR_DIMENSION}-dimensional vectors`);
    logger.info(`  - Vectors will be stored in your vector database (${env.VECTOR_DB_PROVIDER})`);
    logger.info(`  - Similar texts will have similar vector representations`);
    logger.info(`  - You can use these vectors for semantic search and recommendations`);

  } catch (error) {
    logger.error('❌ Embedding model test failed:', error);
    process.exit(1);
  }
};

// Run the test
testEmbedding()
  .then(() => {
    logger.info('\nTest completed. Exiting...');
    process.exit(0);
  })
  .catch((error) => {
    logger.error('Unexpected error:', error);
    process.exit(1);
  });

