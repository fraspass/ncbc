from setuptools import setup

setup(
	name='ncbc',
	version='1.0',
	packages=[
		'ncbc',
	],
	install_requires=[
		'numpy',
		'scipy',
		'gensim',
		'scikit-learn',
		'nltk',
	],
)