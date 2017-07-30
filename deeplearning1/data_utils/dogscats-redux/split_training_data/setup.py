from setuptools import setup

setup(
    name='split_training_data.py',
    version='0.0.2',
    py_modules=['split_training_data'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        split_training_data=split_training_data:cli
    ''',
)