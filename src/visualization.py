import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from .config import PLOTS_DIR

class Visualizer:
    def save_plot(self, fig, filename):
        path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved plot: {path}")

    def plot_sentiment_dist(self, df):
        plt.figure(figsize=(8, 6))
        sns.countplot(x='sentiment', data=df)
        plt.title('Sentiment Distribution')
        self.save_plot(plt.gcf(), 'sentiment_distribution.png')

    def plot_wordcloud(self, text, title):
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        self.save_plot(plt.gcf(), f'wordcloud_{title.lower()}.png')

    def plot_training_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        
        self.save_plot(plt.gcf(), 'training_history.png')
