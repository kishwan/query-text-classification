from src.parameters import parameters
from src.preprocessing import preprocessing
from src.model import model
from src.run import Run

# ADAPTING FROM THIS MODEL https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch
class Controller(parameters):
	
	def __init__(self):
		# Preprocessing pipeline
		self.data = self.prepare_data(parameters.num_words, parameters.in_channels)
		# Initialize the model
		self.model = model(parameters)
		
		# Training - Evaluation pipeline
		Run().train(self.model, self.data, parameters)
		
		
	@staticmethod
	def prepare_data(num_words, in_channels):
		# Preprocessing pipeline
		pr = preprocessing(num_words, in_channels)
		pr.load_data()
		pr.clean_text()
		pr.text_tokenization()
		pr.build_vocabulary()
		pr.word_to_idx()
		pr.padding_sentences()
		pr.split_data()
	
		return {'x_train': pr.x_train, 'y_train': pr.y_train, 'x_test': pr.x_test, 'y_test': pr.y_test}
		
if __name__ == '__main__':
	controller = Controller()