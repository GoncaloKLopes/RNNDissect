from setuptools import setup
setup(name="rnndissect",
      version="0.1",
      description="rnndissect",
      author="Gonçalo Lopes",
      author_email="goncalo.chincho.lopes@ist.utl.pt",
      packages=["rnndissect", "rnndissect.utils", "rnndissect.activations", 
                "rnndissect.activations"],
      zip_safe=False)
