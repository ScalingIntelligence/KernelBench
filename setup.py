from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="kernelbench",
        version="0.2.0",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
    )
