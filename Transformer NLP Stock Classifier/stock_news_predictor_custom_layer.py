import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense

# Positional Encoding Layer
    # 순서 정보 넣어주는 층
    # Keras.Layer 클래스를 상속받은 클래스.
class PositionalEncoding(Layer):

    # 생성자메소드. 클래스 object 생성할 때 호출실행됨.
        # 주로 layer에서 필요한 하이퍼파라미터, 설정들을 초기화함.
    def __init__(self, embed_size, max_len, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        
        # 임베딩차원크기 저장(256).
        self.embed_size = embed_size
        self.max_len = max_len

        # max_len=200. 0~199까지의 값을 생성하면서 + 축을 하나 더 만들어서 2차원으로. (200, 1)
        pos = np.arange(max_len)[:, np.newaxis]
        
        # 0~255까지의 값을 생성하면서 + 축을 하나 더 만들어서 2차원으로. (1, 256)
            # 왜 이걸 embed_size로 결정하지? 그냥 데이터당 들어올 단어 수인 max_len으로 하는게 맞지 않나?
        i = np.arange(embed_size)[np.newaxis, :]

        
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_size))

        # 짝수? 홀수? 일 때 sin, cos 함수 적용
        self.positional_encoding = np.where(i % 2 == 0,
                                            np.sin(pos * angle_rates),
                                            np.cos(pos * angle_rates))
        # 텐서 상수로 처리?
        self.positional_encoding = tf.constant(self.positional_encoding, dtype=tf.float32)

    # 포지셔널임베딩을 더하기
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.positional_encoding[:seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "max_len": self.max_len
        })
        return config


# 셀프어텐션
# # 멀티헤드 어텐션은 이걸 위아래로 쌓은 것.
class SelfAttention(Layer):
    def __init__(self, embed_size, heads, kernel_regularizer=None, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        # 256, 32, 8
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # False면 아래 메시지를 띄우고 실행을 멈춤.
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Q, K, V
        self.values = Dense(self.head_dim, use_bias=False, kernel_regularizer=kwargs.get('kernel_regularizer'))
        self.keys = Dense(self.head_dim, use_bias=False, kernel_regularizer=kwargs.get('kernel_regularizer'))
        self.queries = Dense(self.head_dim, use_bias=False, kernel_regularizer=kwargs.get('kernel_regularizer'))
        # full connected : 마지막 덴스층.
        self.fc_out = Dense(embed_size, kernel_regularizer=kwargs.get('kernel_regularizer'))

    def call(self, values, keys, query, mask):
        N = tf.shape(query)[0]
        value_len, key_len, query_len = tf.shape(values)[1], tf.shape(keys)[1], tf.shape(query)[1]

        values = tf.reshape(values, (N, value_len, self.heads, self.head_dim))
        keys = tf.reshape(keys, (N, key_len, self.heads, self.head_dim))
        queries = tf.reshape(query, (N, query_len, self.heads, self.head_dim))

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # K 부분을 트랜스포즈
        keys_transpose = tf.transpose(keys, perm=[0, 1, 3, 2])  # Transpose for matmul
        # matmul으로 내적
        energy = tf.matmul(queries, keys_transpose)  # (N, heads, query_len, key_len)

        # 마스킹
        if mask is not None:
            energy = tf.where(mask == 0, -1e20, energy)

        # softmax 적용
        attention = tf.nn.softmax(energy / (self.embed_size ** (1 / 2)), axis=-1)

        out = tf.matmul(attention, values)  # (N, heads, query_len, head_dim)
        out = tf.transpose(out, perm=[0, 2, 1, 3])  # Transpose back to (N, query_len, heads, head_dim)
        out = tf.reshape(out, (N, query_len, self.heads * self.head_dim))  # Flatten the last two dimensions
        out = self.fc_out(out)

        # output과 중요도를 담은 어텐션.
        return out, attention

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "heads": self.heads
        })
        return config