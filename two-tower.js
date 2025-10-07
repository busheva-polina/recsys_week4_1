// Two-Tower Recommender Models for MovieLens 100K
// Baseline (Matrix Factorization) and Deep (MLP) versions

class TwoTowerBaseline {
    constructor(numUsers, numItems, embDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        
        // Learnable embedding tables (variables)
        this.userEmbedding = tf.variable(
            tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'userEmbedding'
        );
        this.itemEmbedding = tf.variable(
            tf.randomNormal([numItems, embDim], 0, 0.05), true, 'itemEmbedding'
        );
    }

    userForward(userIdxTensor) {
        // [B,1] int32 -> [B,embDim]
        return tf.gather(this.userEmbedding, userIdxTensor.squeeze([-1]));
    }

    itemForward(itemIdxTensor) {
        // [B,1] int32 -> [B,embDim]
        return tf.gather(this.itemEmbedding, itemIdxTensor.squeeze([-1]));
    }

    score(uEmb, iEmb) {
        // Dot product along last dim with L2 normalization
        const u = tf.l2Normalize(uEmb, -1);
        const v = tf.l2Normalize(iEmb, -1);
        return tf.sum(u.mul(v), -1, true); // [B,1]
    }

    predict(userIdx, itemIdx) {
        return tf.tidy(() => {
            const uEmb = this.userForward(userIdx);
            const iEmb = this.itemForward(itemIdx);
            return this.score(uEmb, iEmb);
        });
    }

    getTrainableVariables() {
        return [this.userEmbedding, this.itemEmbedding];
    }
}

class TwoTowerDeep {
    constructor(numUsers, numItems, embDim, hiddenDim, genreDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        this.hiddenDim = hiddenDim;
        this.genreDim = genreDim;

        // ID embeddings
        this.userIdEmbedding = tf.variable(
            tf.randomNormal([numUsers, embDim], 0, 0.05), true, 'userIdEmbedding'
        );
        this.itemIdEmbedding = tf.variable(
            tf.randomNormal([numItems, embDim], 0, 0.05), true, 'itemIdEmbedding'
        );

        // Project genres -> emb space (learnable)
        this.genreW = tf.variable(
            tf.randomNormal([genreDim, embDim], 0, 0.05), true, 'genreW'
        );

        // User tower MLP layers
        this.userW1 = tf.variable(
            tf.randomNormal([embDim, hiddenDim], 0, 0.05), true, 'userW1'
        );
        this.userB1 = tf.variable(tf.zeros([hiddenDim]), true, 'userB1');
        this.userW2 = tf.variable(
            tf.randomNormal([hiddenDim, embDim], 0, 0.05), true, 'userW2'
        );
        this.userB2 = tf.variable(tf.zeros([embDim]), true, 'userB2');

        // Item tower MLP layers
        this.itemW1 = tf.variable(
            tf.randomNormal([embDim * 2, hiddenDim], 0, 0.05), true, 'itemW1'
        ); // *2 because we concat idEmb + genreEmb
        this.itemB1 = tf.variable(tf.zeros([hiddenDim]), true, 'itemB1');
        this.itemW2 = tf.variable(
            tf.randomNormal([hiddenDim, embDim], 0, 0.05), true, 'itemW2'
        );
        this.itemB2 = tf.variable(tf.zeros([embDim]), true, 'itemB2');
    }

    userForward(userIdx) {
        // [B,1] -> [B,emb]
        return tf.tidy(() => {
            const idEmb = tf.gather(this.userIdEmbedding, userIdx.squeeze([-1]));
            
            // MLP: embDim -> hiddenDim -> embDim
            const h1 = tf.relu(idEmb.matMul(this.userW1).add(this.userB1));
            const out = h1.matMul(this.userW2).add(this.userB2);
            
            return tf.l2Normalize(out, -1);
        });
    }

    itemForward(itemIdx, itemGenresOneHot) {
        // [B,1], [B,G] -> [B,emb]
        return tf.tidy(() => {
            const idEmb = tf.gather(this.itemIdEmbedding, itemIdx.squeeze([-1]));
            const genreEmb = itemGenresOneHot.matMul(this.genreW); // project one-hot genres
            const combined = tf.concat([idEmb, genreEmb], -1); // [B, 2*embDim]
            
            // MLP: 2*embDim -> hiddenDim -> embDim
            const h1 = tf.relu(combined.matMul(this.itemW1).add(this.itemB1));
            const out = h1.matMul(this.itemW2).add(this.itemB2);
            
            return tf.l2Normalize(out, -1);
        });
    }

    score(uEmb, iEmb) {
        // Dot product with L2 normalization
        const u = tf.l2Normalize(uEmb, -1);
        const v = tf.l2Normalize(iEmb, -1);
        return tf.sum(u.mul(v), -1, true); // [B,1]
    }

    predict(userIdx, itemIdx, itemGenres) {
        return tf.tidy(() => {
            const uEmb = this.userForward(userIdx);
            const iEmb = this.itemForward(itemIdx, itemGenres);
            return this.score(uEmb, iEmb);
        });
    }

    getTrainableVariables() {
        return [
            this.userIdEmbedding, this.itemIdEmbedding, this.genreW,
            this.userW1, this.userB1, this.userW2, this.userB2,
            this.itemW1, this.itemB1, this.itemW2, this.itemB2
        ];
    }
}

// Loss functions
class TwoTowerLoss {
    static inBatchSoftmaxLoss(userEmbs, itemEmbs) {
        return tf.tidy(() => {
            // Normalize embeddings
            const u = tf.l2Normalize(userEmbs, -1); // [B, D]
            const v = tf.l2Normalize(itemEmbs, -1); // [B, D]
            
            // Compute logits: U @ V^T -> [B, B]
            const logits = u.matMul(v.transpose());
            
            // Labels: diagonal positions are positive (identity matrix)
            const batchSize = userEmbs.shape[0];
            const labels = tf.oneHot(tf.range(0, batchSize), batchSize);
            
            // Softmax cross entropy
            const loss = tf.losses.softmaxCrossEntropy(labels, logits);
            return loss;
        });
    }

    static bprLoss(userEmbs, posItemEmbs, negItemEmbs) {
        return tf.tidy(() => {
            // Normalize embeddings
            const u = tf.l2Normalize(userEmbs, -1);
            const pos = tf.l2Normalize(posItemEmbs, -1);
            const neg = tf.l2Normalize(negItemEmbs, -1);
            
            // Compute scores
            const posScores = tf.sum(u.mul(pos), -1); // [B]
            const negScores = tf.sum(u.mul(neg), -1); // [B]
            
            // BPR loss: -log σ(pos_score - neg_score)
            const diff = posScores.sub(negScores);
            const loss = tf.mean(tf.neg(tf.logSigmoid(diff)));
            
            return loss;
        });
    }
}

// Training utilities
class TwoTowerTrainer {
    constructor(model, optimizer, lossType = 'softmax') {
        this.model = model;
        this.optimizer = optimizer;
        this.lossType = lossType;
    }

    async trainStep(userBatch, itemBatch, genresBatch = null) {
        return tf.tidy(() => {
            const lossFunction = () => {
                if (this.lossType === 'softmax') {
                    const userEmbs = this.model.userForward(userBatch);
                    const itemEmbs = genresBatch ? 
                        this.model.itemForward(itemBatch, genresBatch) : 
                        this.model.itemForward(itemBatch);
                    return TwoTowerLoss.inBatchSoftmaxLoss(userEmbs, itemEmbs);
                } else { // BPR
                    // For BPR, we need negative samples - use in-batch negatives
                    const userEmbs = this.model.userForward(userBatch);
                    const posItemEmbs = genresBatch ? 
                        this.model.itemForward(itemBatch, genresBatch) : 
                        this.model.itemForward(itemBatch);
                    
                    // Shuffle items for negatives
                    const negIndices = tf.util.createShuffledIndices(itemBatch.shape[0]);
                    const negItemBatch = tf.gather(itemBatch, negIndices);
                    const negGenresBatch = genresBatch ? tf.gather(genresBatch, negIndices) : null;
                    
                    const negItemEmbs = negGenresBatch ? 
                        this.model.itemForward(negItemBatch, negGenresBatch) : 
                        this.model.itemForward(negItemBatch);
                    
                    return TwoTowerLoss.bprLoss(userEmbs, posItemEmbs, negItemEmbs);
                }
            };

            const loss = this.optimizer.minimize(lossFunction, true, this.model.getTrainableVariables());
            return loss ? loss.dataSync()[0] : 0;
        });
    }
}
