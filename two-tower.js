// Two-Tower Recommender Models for MovieLens 100K

class TwoTowerBaseline {
    constructor(numUsers, numItems, embDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        
        // Learnable embedding tables
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
        // Normalize embeddings and compute dot product
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

        // Project genres -> emb space
        this.genreW = tf.variable(
            tf.randomNormal([genreDim, embDim], 0, 0.05), true, 'genreW'
        );

        // User tower MLP weights
        this.userW1 = tf.variable(
            tf.randomNormal([embDim, hiddenDim], 0, 0.05), true, 'userW1'
        );
        this.userB1 = tf.variable(tf.zeros([hiddenDim]), true, 'userB1');
        this.userW2 = tf.variable(
            tf.randomNormal([hiddenDim, embDim], 0, 0.05), true, 'userW2'
        );
        this.userB2 = tf.variable(tf.zeros([embDim]), true, 'userB2');

        // Item tower MLP weights
        this.itemW1 = tf.variable(
            tf.randomNormal([embDim * 2, hiddenDim], 0, 0.05), true, 'itemW1'
        );
        this.itemB1 = tf.variable(tf.zeros([hiddenDim]), true, 'itemB1');
        this.itemW2 = tf.variable(
            tf.randomNormal([hiddenDim, embDim], 0, 0.05), true, 'itemW2'
        );
        this.itemB2 = tf.variable(tf.zeros([embDim]), true, 'itemB2');
    }

    userForward(userIdx) {
        return tf.tidy(() => {
            const idEmb = tf.gather(this.userIdEmbedding, userIdx.squeeze([-1]));
            
            // MLP: embDim -> hiddenDim -> embDim
            const h1 = tf.relu(idEmb.matMul(this.userW1).add(this.userB1));
            const out = h1.matMul(this.userW2).add(this.userB2);
            
            return tf.l2Normalize(out, -1);
        });
    }

    itemForward(itemIdx, itemGenresOneHot) {
        return tf.tidy(() => {
            const idEmb = tf.gather(this.itemIdEmbedding, itemIdx.squeeze([-1]));
            const genreEmb = itemGenresOneHot.matMul(this.genreW);
            const combined = idEmb.concat(genreEmb, -1);
            
            // MLP: (embDim*2) -> hiddenDim -> embDim
            const h1 = tf.relu(combined.matMul(this.itemW1).add(this.itemB1));
            const out = h1.matMul(this.itemW2).add(this.itemB2);
            
            return tf.l2Normalize(out, -1);
        });
    }

    score(uEmb, iEmb) {
        const u = tf.l2Normalize(uEmb, -1);
        const v = tf.l2Normalize(iEmb, -1);
        return tf.sum(u.mul(v), -1, true);
    }

    predict(userIdx, itemIdx, itemGenresOneHot) {
        return tf.tidy(() => {
            const uEmb = this.userForward(userIdx);
            const iEmb = this.itemForward(itemIdx, itemGenresOneHot);
            return this.score(uEmb, iEmb);
        });
    }
}

// Loss functions
class LossFunctions {
    static inBatchSoftmaxLoss(userEmbs, itemEmbs, temperature = 1.0) {
        return tf.tidy(() => {
            // Normalize embeddings
            const u = tf.l2Normalize(userEmbs, -1);
            const v = tf.l2Normalize(itemEmbs, -1);
            
            // Compute similarity matrix: [B, B]
            const logits = tf.matMul(u, v, false, true).div(tf.scalar(temperature));
            
            // Labels are diagonal (each user matches with corresponding item)
            const batchSize = userEmbs.shape[0];
            const labels = tf.oneHot(tf.range(0, batchSize), batchSize);
            
            // Softmax cross entropy
            const losses = tf.softmaxCrossEntropy(labels, logits);
            return tf.mean(losses);
        });
    }

    static bprLoss(userEmbs, posItemEmbs, negItemEmbs) {
        return tf.tidy(() => {
            const u = tf.l2Normalize(userEmbs, -1);
            const pos = tf.l2Normalize(posItemEmbs, -1);
            const neg = tf.l2Normalize(negItemEmbs, -1);
            
            const posScores = tf.sum(u.mul(pos), -1);
            const negScores = tf.sum(u.mul(neg), -1);
            
            // BPR: -log σ(pos_score - neg_score)
            const diff = posScores.sub(negScores);
            const losses = tf.softplus(tf.neg(diff)); // -log(sigmoid(diff))
            
            return tf.mean(losses);
        });
    }
}

// Training utilities
class TrainingUtils {
    static createOptimizer(learningRate) {
        return tf.train.adam(learningRate);
    }

    static sampleNegativeItems(positiveItems, numItems, numNegatives) {
        const negatives = [];
        for (let i = 0; i < numNegatives; i++) {
            let neg;
            do {
                neg = Math.floor(Math.random() * numItems);
            } while (positiveItems.has(neg));
            negatives.push(neg);
        }
        return negatives;
    }
}
