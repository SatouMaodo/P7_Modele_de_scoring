from setuptools import setup, find_packages

setup(
    name='mon_application_mlflow',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'mlflow',
        'gunicorn',
        'pytest',
        # Ajoutez ici les autres dépendances de votre application
    ],
    entry_points={
        'console_scripts': [
            'demarrer_mlflow=mon_application_mlflow.demarrer:main',
        ],
    },
)
