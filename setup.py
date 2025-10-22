from setuptools import setup, find_packages

setup(
    name='traffic-sign-recognition',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'pillow',
        'matplotlib',
        'scikit-learn',
        'streamlit',
    ],
)
