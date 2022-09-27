import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='classy',
    version='1.0.0',
    author='nlpander',
    author_email='tarafulis@gmail.com',
    description='a package to do distributed preprocessing and training of neural networks for classification',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/nlpander/classy',
    project_urls = {
        "Bug Tracker": "https://github.com/nlpander/classy/issues"
    },
    license ='GNU',
    #packages = setuptools.find_packages(),
    packages = ['classy','classy.preprocess','classy.train'],
    install_requires=['ray==1.13.0','numpy','pandas',
                    'torch==1.11.0','nltk==3.7',spacy,
                    'gensim==4.1.2','matplotlib',
                    'datasets',],
)

#'spacy==3.3.1'