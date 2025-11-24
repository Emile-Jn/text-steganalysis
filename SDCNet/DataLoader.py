import re
import os
import random


class Example:
	"""Minimal example object with text and label attributes.
	Used to replace torchtext.data.Example for a lightweight implementation.
	"""
	def __init__(self, text, label):
		self.text = text
		self.label = label


class MyData:
	@staticmethod
	def sort_ket(ex):
		return len(ex.text)

	# alias for callers expecting sort_key
	sort_key = sort_ket

	def __init__(self, text_field, label_field, cover_path=None,
				 stego_path=None, examples=None, **kwargs):
		"""Create a lightweight dataset-like object.

		Arguments:
			text_field: callable or object with .preprocess(text) -> processed_text.
			label_field: callable or object with .preprocess(label) -> processed_label.
			cover_path/stego_path: paths to files containing one sample per line.
			examples: optional list of Example instances to initialize from.
		"""

		def clean_str(string):
			"""
			Tokenization/string cleaning (from original implementation).
			"""
			string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
			string = re.sub(r"\'s", " \'s", string)
			string = re.sub(r"\'ve", " \'ve", string)
			string = re.sub(r"n\'t", " n\'t", string)
			string = re.sub(r"\'re", " \'re", string)
			string = re.sub(r"\'d", " \'d", string)
			string = re.sub(r"\'ll", " \'ll", string)
			string = re.sub(r",", " , ", string)
			string = re.sub(r"!", " ! ", string)
			string = re.sub(r"\(", " \( ", string)
			string = re.sub(r"\)", " \) ", string)
			string = re.sub(r"\?", " \? ", string)
			string = re.sub(r"\s{2,}", " ", string)
			return string.strip()

		self.text_field = text_field
		self.label_field = label_field

		# Normalize preprocessors: prefer callable, else look for .preprocess attr
		if callable(text_field):
			self._text_preproc = text_field
		elif hasattr(text_field, 'preprocess') and callable(getattr(text_field, 'preprocess')):
			self._text_preproc = text_field.preprocess
		else:
			# fallback uses clean_str only
			self._text_preproc = lambda s: clean_str(s)

		if callable(label_field):
			self._label_preproc = label_field
		elif hasattr(label_field, 'preprocess') and callable(getattr(label_field, 'preprocess')):
			self._label_preproc = label_field.preprocess
		else:
			self._label_preproc = lambda s: s

		self.examples = []

		if examples is not None:
			# assume list of Example-like objects or (text,label) tuples
			for ex in examples:
				if isinstance(ex, Example):
					self.examples.append(ex)
				elif isinstance(ex, (list, tuple)) and len(ex) >= 2:
					text = ex[0]
					label = ex[1]
					self.examples.append(Example(self._text_preproc(text), self._label_preproc(label)))
				else:
					# try to read dict-like
					text = getattr(ex, 'text', None) or ex.get('text') if isinstance(ex, dict) else None
					label = getattr(ex, 'label', None) or ex.get('label') if isinstance(ex, dict) else None
					if text is not None and label is not None:
						self.examples.append(Example(self._text_preproc(text), self._label_preproc(label)))
					else:
						raise ValueError('Invalid example format in examples list')
		else:
			# load from files if provided
			if cover_path is None or stego_path is None:
				raise ValueError('cover_path and stego_path must be provided when examples is None')

			if not os.path.isfile(cover_path):
				raise FileNotFoundError(f'Cover file not found: {cover_path}')
			if not os.path.isfile(stego_path):
				raise FileNotFoundError(f'Stego file not found: {stego_path}')

			with open(cover_path, 'r', errors='ignore') as f:
				for line in f:
					line = line.rstrip('\n')
					if not line:
						continue
					proc = self._text_preproc(clean_str(line))
					self.examples.append(Example(proc, self._label_preproc('negative')))

			with open(stego_path, 'r', errors='ignore') as f:
				for line in f:
					line = line.rstrip('\n')
					if not line:
						continue
					proc = self._text_preproc(clean_str(line))
					self.examples.append(Example(proc, self._label_preproc('positive')))

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		return self.examples[idx]

	@classmethod
	def split(cls, text_field, label_field, args, state,
				 shuffle=True, **kwargs):
		"""Split factory similar to original: for 'train' returns (train, val), for 'test' returns dataset.

		Arguments:
			args: object with attributes train_cover_dir, train_stego_dir, test_cover_dir, test_stego_dir
			state: 'train' or 'test'
		"""
		if state == 'train':
			print('loading the training data...')
			cover_path = args.train_cover_dir
			stego_path = args.train_stego_dir
			ds = cls(text_field, label_field, cover_path=cover_path, stego_path=stego_path)
			examples = ds.examples
			if shuffle:
				random.shuffle(examples)
			val_idx = -2000
			return (cls(text_field, label_field, examples=examples[:val_idx]),
					cls(text_field, label_field, examples=examples[val_idx:]))

		if state == 'test':
			print('loading the testing data...')
			cover_path = args.test_cover_dir
			stego_path = args.test_stego_dir
			return cls(text_field, label_field, cover_path=cover_path, stego_path=stego_path)

		raise ValueError("Unknown state: expected 'train' or 'test'")
