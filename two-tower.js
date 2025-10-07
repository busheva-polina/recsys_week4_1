class TwoTowerBaseline {
    constructor(numUsers, numItems, embDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        
        // Learnable embedding tables
        this.userEmbedding = tf.variable(
            tf.randomNormal([numUsers, embDim], 0, 0.05), 
            true, 
            'userEmbedding'
        );
        
        this.itemEmbedding = tf.variable(
            tf.randomNormal([numItems, embDim], 0, 0.05), 
            true, 
            'itemEmbedding'
        );
    }
    
    userForward(userIdxTensor) {
        // userIdxTensor: [B] or [B,1] int32 -> returns [B, embDim]
        const squeezed = userIdxTensor.squeeze([-1]);
        return tf.gather(this.userEmbedding, squeezed);
    }
    
    itemForward(itemIdxTensor) {
        // itemIdxTensor: [B] or [B,1] int32 -> returns [B, embDim]
        const squeezed = itemIdxTensor.squeeze([-1]);
        return tf.gather(this.itemEmbedding, squeezed);
    }
    
    score(uEmb, iEmb) {
        // Compute normalized dot product (cosine similarity)
        const u = tf.l2Normalize(uEmb, -1);
        const v = tf.l2Normalize(iEmb, -1);
        return tf.sum(u.mul(v), -1, true); // [B, 1]
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
            tf.randomNormal([numUsers, embDim], 0, 0.05), 
            true, 
            'userIdEmbedding'
        );
        
        this.itemIdEmbedding = tf.variable(
            tf.randomNormal([numItems, embDim], 0, 0.05), 
            true, 
            'itemIdEmbedding'
        );
        
        // Genre projection matrix
        this.genreW = tf.variable(
            tf.randomNormal([genreDim, embDim], 0, 0.05), 
            true, 
            'genreW'
        );
        
        // Initialize MLP weights
        this.initializeMLPWeights();
    }
    
    initializeMLPWeights() {
        // User tower MLP: [embDim] -> hiddenDim -> embDim
        this.userW1 = tf.variable(
            tf.randomNormal([this.embDim, this.hiddenDim], 0, 0.05), 
            true, 
            'userW1'
        );
        this.userB1 = tf.variable(tf.zeros([this.hiddenDim]), true, 'userB1');
        this.userW2 = tf.variable(
            tf.randomNormal([this.hiddenDim, this.embDim], 0, 0.05), 
            true, 
            'userW2'
        );
        this.userB2 = tf.variable(tf.zeros([this.embDim]), true, 'userB2');
        
        // Item tower MLP: [embDim * 2] -> hiddenDim -> embDim
        // (item_id_emb + genre_emb = 2 * embDim)
        this.itemW1 = tf.variable(
            tf.randomNormal([this.embDim * 2, this.hiddenDim], 0, 0.05), 
            true, 
            'itemW1'
        );
        this.itemB1 = tf.variable(tf.zeros([this.hiddenDim]), true, 'itemB1');
        this.itemW2 = tf.variable(
            tf.randomNormal([this.hiddenDim, this.embDim], 0, 0.05), 
            true, 
            'itemW2'
        );
        this.itemB2 = tf.variable(tf.zeros([this.embDim]), true, 'itemB2');
    }
    
    userForward(userIdxTensor) {
        return tf.tidy(() => {
            const squeezed = userIdxTensor.squeeze([-1]);
            const idEmb = tf.gather(this.userIdEmbedding, squeezed);
            return this.mlp(idEmb, 'user');
        });
    }
    
    itemForward(itemIdxTensor, itemGenresOneHot) {
        return tf.tidy(() => {
            const squeezed = itemIdxTensor.squeeze([-1]);
            const idEmb = tf.gather(this.itemIdEmbedding, squeezed);
            const genreEmb = itemGenresOneHot.matMul(this.genreW);
            
            // Concatenate ID embedding and genre embedding
            const combined = tf.concat([idEmb, genreEmb], -1);
            return this.mlp(combined, 'item');
        });
    }
    
    mlp(x, tower) {
        return tf.tidy(() => {
            let W1, b1, W2, b2;
            
            if (tower === 'user') {
                W1 = this.userW1;
                b1 = this.userB1;
                W2 = this.userW2;
                b2 = this.userB2;
            } else {
                W1 = this.itemW1;
                b1 = this.itemB1;
                W2 = this.itemW2;
                b2 = this.itemB2;
            }
            
            // Hidden layer with ReLU
            const h1 = x.matMul(W1).add(b1).relu();
            
            // Output layer (normalized)
            const out = h1.matMul(W2).add(b2);
            return tf.l2Normalize(out, -1);
        });
    }
    
    score(uEmb, iEmb) {
        // Compute normalized dot product (cosine similarity)
        const u = tf.l2Normalize(uEmb, -1);
        const v = tf.l2Normalize(iEmb, -1);
        return tf.sum(u.mul(v), -1, true); // [B, 1]
    }
}
