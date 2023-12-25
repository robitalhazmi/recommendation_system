from flask import Flask, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import keras
from keras import layers
from keras import ops

app = Flask(__name__)
CORS(app)

rating_df = pd.read_csv('./datasets/rating_history.csv')
product_df = pd.read_csv('./datasets/product_details.csv')

user_ids = rating_df["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
product_ids = rating_df["product_id"].unique().tolist()
product2product_encoded = {x: i for i, x in enumerate(product_ids)}
product_encoded2product = {i: x for i, x in enumerate(product_ids)}

num_users = len(user2user_encoded)
num_products = len(product_encoded2product)

EMBEDDING_SIZE = 50

class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_products, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_products = num_products
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.product_embedding = layers.Embedding(
            num_products,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.product_bias = layers.Embedding(num_products, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        product_vector = self.product_embedding(inputs[:, 1])
        product_bias = self.product_bias(inputs[:, 1])
        dot_user_product = ops.tensordot(user_vector, product_vector, 2)
        # Add all the components (including bias)
        x = dot_user_product + user_bias + product_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return ops.nn.sigmoid(x)

model = RecommenderNet(num_users, num_products, EMBEDDING_SIZE)
model.load_weights("./model/recommendation_model.weights.h5")

def neural_collaborative_filtering(user_id):
    products_rated_by_user = rating_df[rating_df['user_id'] == user_id]
    products_not_rated = product_df[~product_df["product_id"].isin(products_rated_by_user['product_id'].values)]["product_id"]
    products_not_rated = list(set(products_not_rated).intersection(set(product2product_encoded.keys())))
    products_not_rated = [[product2product_encoded.get(x)] for x in products_not_rated]
    user_encoder = user2user_encoded.get(user_id)
    user_product_array = np.hstack(([[user_encoder]] * len(products_not_rated), products_not_rated))

    ratings = model(user_product_array, training=False).numpy().flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_product_ids = [product_encoded2product.get(products_not_rated[x][0]) for x in top_ratings_indices]
    recommended_products = product_df[product_df["product_id"].isin(recommended_product_ids)]
    
    return recommended_products

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Ambil data dari form jika metode adalah POST
        customer_id = request.form.get('customerID')
        
        # Panggil model untuk mendapatkan rekomendasi produk
        recommendations = neural_collaborative_filtering(customer_id)
        product_name = list(recommendations['product_name'].values)
        img_link = list(recommendations['img_link'].values)
        
        return render_template('index.html', result={
            'product_name': product_name,
            'img_link': img_link,
            'customer_id': customer_id
        })
    else:
        # Jika metode adalah GET, tampilkan halaman utama
        return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)