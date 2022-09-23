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
    packages = setuptools.find_packages(),
    install_requires=['ray==1.13.0','numpy==1.22.4','pandas==1.4.2',
                    'torch==1.12.1+cu116','nltk==3.7',
                    'gensim==4.1.2','spacy==3.3.1','matplotlib',
                    'datasets',],
)