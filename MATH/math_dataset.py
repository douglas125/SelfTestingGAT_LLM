"""
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""
import os
import json
import tarfile
import requests

import numpy as np
from tqdm import tqdm

rng = np.random.default_rng()

def get_MATH_dataset():
    math_path = 'MATH/MATH.tar'
    if not os.path.isfile(math_path):
        url = 'https://people.eecs.berkeley.edu/~hendrycks/MATH.tar'
        r = requests.get(url, allow_redirects=True)
        open(math_path, 'wb').write(r.content)

    tar = tarfile.open("MATH/MATH.tar")
    math_dataset = {
        'test': {},
        'train': {},
    }
    for member in tqdm(tar.getmembers()):
        f = tar.extractfile(member)
        if f is not None and member.name.endswith('.json'):
            name_split = member.name.split('/')[1:]
            cur_cat = name_split[1]
            cur_list = math_dataset[name_split[0]].get(cur_cat, [])

            content = f.read()
            cur_list.append(json.loads(content))
            math_dataset[name_split[0]][cur_cat] = cur_list

    tar.close()
    return math_dataset


def get_random_sample(math_dataset, chosen_set='train'):
    categs = math_dataset[chosen_set].keys()
    random_categ = rng.choice(list(categs))
    sample = rng.choice(math_dataset[chosen_set][random_categ])
    sample['sol_text'] = f"""The expected answer is:
<expected_answer>
{sample['solution']}
</expected_answer>

Consider only the final result marked by $\\boxed
Did your code produce the correct answer? Answer with YES or NO inside the <answer_was_correct> tag.
"""
    return sample
