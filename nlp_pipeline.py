from sklearn.tree import DecisionTreeClassifier as dtc 
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import cross_val_score as cv_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from collections import defaultdict
from textblob import TextBlob
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score as accuracy
from lemmatizer import Lemmatizer


class NLPPipeline():
	def __init__(self, text, Y, train_size=.85):
        text = text.apply(lambda review: review.replace(',', '.'))  # added this
		self.model_builders = {'dtc': dtc, 'rfc': rfc}
		steps = ['tfidf', 'feature_engineering', 'lda', 'model']
		self.pipeline_dic = {step: None for step in steps}
		self.text_train, self.text_test, self.Y_train, self.Y_test = split(text, Y, train_size=train_size, stratify=Y)
		self.keep_tfidf = lambda tfidf_dic: (tfidf_dic == self.pipeline_dic['tfidf'])
		self.keep_features = lambda features_dic: (features_dic == self.pipeline_dic['features'])
		self.prob_info = lambda prob: -prob * np.log(prob)
		self.pipeline_dic = {step: "Default" for step in steps}
		self.train_size = train_size

	def update_tfidf(self, tfidf_dic):
        tfidf_dic['tokenizer'] = Lemmatizer() #added this line
		self.pipeline_dic['tfidf'] = tfidf_dic
		self.tfidf = TfidfVectorizer(**tfidf_dic) #implement tokenizer/lemmatizer here?
		self.tfidf_train = self.tfidf.fit_transform(self.text_train) 
		self.tfidf_train = self.tfidf_train.toarray()
		#self.tokenizer = self.tfidf.build_tokenizer()
		self.tfidf_test = self.tfidf.transform(self.text_test)
		self.tfidf_test = self.tfidf_test.toarray()
		self.feature_names = self.tfidf.get_feature_names()

	def update_lda(self, lda_dic):
		def calc_topics_words(num_top_words):
			topics_words = []

			for ix, topic in enumerate(self.lda.components_):
				top_word_inds = topic.argsort()[:-num_top_words - 1:-1]
				topic_words = set([self.feature_names[i] for i in top_word_inds])
				topics_words.append(topic_words)

			return topics_words

		num_top_words = lda_dic['num_top_words'] if 'num_top_words' in lda_dic else 10
		lda_model_dic = {k: v for k, v in lda_dic.items() if k!= 'num_top_words'}
		self.lda = LDA(**lda_model_dic)
		self.lda.fit_transform(self.tfidf_train)
		self.topics_words = calc_topics_words(num_top_words)

	def calc_entropy(self, text):
		''' Many other equivalent ways to calculate entropy. This seems to be the fastest. 5 x faster than scipy's entropy method.'''

		word_counts = defaultdict(int)
		text_size = float(len(text))

		for word in text:
			word_counts[word] += 1

		word_counts = np.array(list(word_counts.values()))
		word_probs = word_counts / text_size
		entropy = -1 * sum(map(self.prob_info, word_probs))

		return entropy

	def calc_lda_features(self, tokenized_text):
		num_topics = len(self.topics_words)
		unique_words = set(tokenized_text)
		num_unique_words = float(len(unique_words))
		lda_features = [len(unique_words.intersection(topic_words))
							   / num_unique_words for topic_words in self.topics_words]

		return lda_features

	def calc_sentiment_features(self, text):
		min_polarity, max_polarity = -.1, .1
		blob = TextBlob(text)
		polarities = [sentence.sentiment.polarity for sentence in blob.sentences]
		polarities = [round(polarity, 1) for polarity in polarities]
		polarity_entropy = self.calc_entropy(polarities)
		polarity_var = np.var(polarities)
		num_pos_sents = len([polarity for polarity in polarities if polarity > max_polarity])
		num_neg_sents = len([polarity for polarity in polarities if polarity < min_polarity])
		num_sents = float(len(polarities))

		pos_sent_freq, neg_sent_freq = num_pos_sents / num_sents, num_neg_sents/num_sents
		num_neutral_sents = num_sents - num_pos_sents - num_neg_sents
		max_pol, min_pol= np.max(polarities) if polarities else 0, min(polarities) if polarities else 0
		subjectivities = [sentence.sentiment.subjectivity for sentence in blob.sentences]
		subjectivities = [round(x, 2) for x in subjectivities]
		subj_var = np.var(subjectivities)
		max_subj, min_subj = np.max(subjectivities) if polarities else 0, min(subjectivities) if polarities else 0
		sentiment_features = [polarity_entropy, polarity_var, num_pos_sents, num_neg_sents, num_neutral_sents, pos_sent_freq, neg_sent_freq,
		                  	  num_sents, max_pol, min_pol, subj_var, max_subj, min_subj]

		return sentiment_features

	def update_features(self, features_dic):
		"""
		From a dictionary containing parameter labels and values used for building features (currently just LDA),
		updates feature matrices by re-calculating features for each text. 
		
		Arguments
		features_dic (dictionary): A dictionary with string parameter names as keys and ints/floats as values. 

		Example: 
		features_dic = {'n_components': 10, 'n_words': 10}
		"""

		def calc_features(text):
			words = self.tokenizer(text)
			entropy = self.calc_entropy(words)
			lda_features = self.calc_lda_features(words)
			sentiment_features = self.calc_sentiment_features(text)
			features = [entropy, *lda_features, *sentiment_features]

			return features

		self.pipeline_dic['features'] = features_dic
		self.update_lda(features_dic)
		self.X_train = np.hstack((self.tfidf_train, np.array([np.array(calc_features(text)) for text in self.text_train])))
		self.X_test = np.hstack((self.tfidf_test, np.array([np.array(calc_features(text)) for text in self.text_test])))

	def grid_search(self, step_grids):
		"""
		From a nested dictionary containing grids for each pipeline step, fit and score each possible pipeline
		permutation (nested permutation of the step permutations).
		
		Arguments
		step_grids: A nested dictionary containing the step grid for each step. 
					Example: step_grids = {'tfidf' = {'min_df': [0.1]},
			   							   'features' = {'n_components': [10], num_top_words: [10]},
			   							   'model' = {'type': ['rfc']} }

		Returns
		pipeline_scores: A sorted list of 2-tuples containing the pipeline dictionary and score of each pipeline permutation. 
		"""

		def get_step_perms(grid):
			"""
			From grid (dict) mapping each parameter name to a list of values for that parameter,
			returns the list of all permutations (dicts) that can be made by choosing a different value
			for each parameter from its values list.
			
			Arguments 
			grid ({string: list}): A dictionary mapping parameter names to a list of parameter values. 
								   Example: grid = {'min_df': [0.1], 'max_df': [0.8, 0.9]}

			Returns 
			step_perms ([dict]): A list of all dictionary permutations for the step that can be made 
								 by choosing different parameter values from each parameter's domain. 

								 Example: For the above grid example, we'd have
								 step_perms = [{'min_df': 0.1, 'max_df: 0.8'}, {'min_df: 0.1', max_df: 0.9}]

			"""

			param_names = list(grid.keys())
			param_val_perms = list(product(*list(grid.values())))
			num_params = len(param_names)
			step_perms = [{param_names[j]: param_val_perm[j] for j in range(num_params)} for param_val_perm in param_val_perms]

			return step_perms

		steps = list(step_grids.keys())
		num_steps = len(steps)
		grids = list(step_grids.values())
		step_perms = list(map(get_step_perms, grids))
		pipeline_perms = list(product(*step_perms))
		pipeline_perms = [{steps[i]: pipeline_perm[i] for i in range(num_steps)} for pipeline_perm in pipeline_perms]
		pipeline_scores = [[pipeline_perm, round(self.score(pipeline_perm), 3)] for pipeline_perm in pipeline_perms]
		pipeline_scores.sort(key=lambda x: x[1], reverse=True)

		return pipeline_scores

	def score(self, pipeline_dic):
		tfidf_vectorizer = TfidfVectorizer(**pipeline_dic['tfidf'])
		keep_tfidf = self.keep_tfidf(pipeline_dic['tfidf'])

		if not keep_tfidf:
			self.update_tfidf(pipeline_dic['tfidf'])

		keep_features = keep_tfidf and self.keep_features(pipeline_dic['features'])

		if not keep_features:
			self.update_features(pipeline_dic['features'])

		self.model_builder = self.model_builders[pipeline_dic['model']['type']]
		model_dic = {key: value for key, value in pipeline_dic['model'].items() if key != 'type'}
		self.model = self.model_builder(**model_dic)
		self.model.fit(self.X_train, self.Y_train)
        score = np.mean(cv_score(self.model, X_test, Y_test, cv=5, scoring = 'accuracy'))
		print(f"Params = {pipeline_dic}, score = {round(score, 3)}. \n")

		return score





