# 📚 Reddit Upvote Model with Deep Learning
- Document here the project: Upvote_Model
- Description: The purpose of this model is to predict the upvotes of posts in [Reddit DogPictures](https://www.reddit.com/r/dogpictures/). The Model maximizes upvotes mainly based on the Image, Title, and Time of a post. Unlike Instagram, Reddit has no celebrities, so the number of followers for each user would not impact so much to the upvotes. Our model hence ignore the number of followers in our prediction model. The model architecture is shown by the graph below.


- Data Source:
- Type of analysis:


Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for Upvote_Model in github.com/{group}. If your project is not set please add it:

Create a new project on github.com/{group}/Upvote_Model
Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "Upvote_Model"
git remote add origin git@github.com:{group}/Upvote_Model.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
Upvote_Model-run
```

# Install

Go to `https://github.com/{group}/Upvote_Model` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/Upvote_Model.git
cd Upvote_Model
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
Upvote_Model-run
```
