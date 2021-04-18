Abstract to Title
=================

This project was completed as part of a university project for a Masters of Datascience class. As part of this project I had to solve a real world problem using an NLP model. I chose to take a "meta" approach and build an NLP model to suggest paper "titles" from their "abstracts". I have designed a docker stack for use in production.

Usage
-----

You will need to clone the repo, set up a virtual environment and then start the API.

```bash
# clone the repo
git clone https://github.com/schlerp/abstract_to_title.git

# cd into the local repo
cd abstract_to_title

# setup a virtual environment
python -m venv venv

# source the virtual environment
source ./venv/bin/activate

# run the API
sh ./start_api.sh
```

Use something like "Postman" to post an abstract to the "/get_title_from_abstract" endpoint.

Author
------

* [Patrick Coffey](https://github.com/schlerp)
