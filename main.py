from src.parameters import parameters
from src.preprocessing import preprocessing
from src.model import model
from src.run import Run
		

class Controller(parameters):
	
	def __init__(self):
		# Preprocessing pipeline
		self.data = self.prepare_data(parameters.num_words, parameters.seq_len)
		
		# Initialize the model
		self.model = model(parameters)
		
		# Training - Evaluation pipeline
		Run().train(self.model, self.data, parameters)
		
		
	@staticmethod
	def prepare_data(num_words, seq_len):
		# Preprocessing pipeline
		pr = preprocessing(num_words, seq_len)
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