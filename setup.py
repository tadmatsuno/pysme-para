import setuptools

setuptools.setup(
    name='pysme_para',
    version='0.0.1',
    description='pysme-para.',
    url='https://github.com/MingjieJian/pysme-para.git',
    author='Mingjie Jian',
    author_email='jian-mingjie@outlook.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Framework :: IPython",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    python_requires=">=3.5",
    install_requires=[

    ],
    include_package_data=True,  
    #   package_data={'': ['moog_nosm/moog_nosm_FEB2017/']},
    zip_safe=False)