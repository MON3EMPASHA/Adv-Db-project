import express from 'express';
import cors from 'cors';
import movieRoutes from './routes/movieRoutes';
import { errorHandler } from './middleware/errorHandler';

const app = express();

app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

app.use('/api', movieRoutes);

app.use(errorHandler);

export default app;





