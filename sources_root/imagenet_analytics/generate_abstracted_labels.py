import json
import nltk
from nltk.corpus import wordnet as wn

def get_synset(id, name):
    offset = int(id[1:])
    for i in range(10):
        synset = wn.synset(f"{name}.n.0{i}")
        if synset._offset == offset:
            return synset

# nltk.download('wordnet')
def load_imagenet_mapping(json_path):
    with open(json_path, 'r') as f:
        class_index = json.load(f)
    return {int(k): get_synset(v[0], v[1])for k, v in class_index.items()}


# get current file path
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))

imagenet_mapping = load_imagenet_mapping(os.path.join(current_file_path, 'imagenet_class_index.json'))


def get_all_hypernyms(synset):
    hypernyms = []
    if len(synset.hypernyms()) > 1:
        print(str(synset.hypernyms()) + "=====")
    for hyper in synset.hypernyms():
        hypernyms.append(hyper)
        all_hypernyms = get_all_hypernyms(hyper)
        hypernyms.extend(all_hypernyms)
    return hypernyms


from collections import defaultdict

# 初始化父类字典
parent_dict = defaultdict(list)

for idx, synset in imagenet_mapping.items():
    # 获取当前synset的所有父类
    all_hypernyms = get_all_hypernyms(synset)
    all_hypernyms = [synset] + all_hypernyms
    print(all_hypernyms)
    # 将子类ID添加到每个父类条目中
    for hyper in all_hypernyms:
        parent_name = hyper.name() # .split('.')[0] # + ".n.01"
        parent_dict[parent_name].append(idx)
del_n = []
for name, ids in parent_dict.items():
    for name2, id2, in parent_dict.items():
        if name < name2 and set(ids) == (set(id2)):
            del_n.append(name)

for name in del_n:
    if name in parent_dict:
        del parent_dict[name]

parent_list = [(name, sorted(list(set(ids)))) for name, ids in parent_dict.items()]

parent_list = sorted(parent_list, key=lambda x: len(x[1]), reverse=True)
print(len(parent_list))
print(parent_list)
print([(n[0], len(n[1])) for n in parent_list])
print([n[0] for n in parent_list])
# select len(n[1] in [20, 100]
parent_list = [n for n in parent_list if 4 <= len(n[1]) <= 999]

print([(n[0], len(n[1])) for n in parent_list])

existed_names = set([n[0] for n in parent_list])

print(existed_names)
# removed:
# Difficult Re-Filtered: Ape or Monkey
# Abstracted and less informative entity, object, machine
unfamiliar_names = {'entity.n.01', 'physical_entity.n.01', 'object.n.01', 'ungulate.n.01', 'whole.n.02', 'animal.n.01','organism.n.01','vertebrate.n.01','vascular_plant.n.01',
                    'instrumentality.n.03', 'mammal.n.01', 'placental.n.01', 'carnivore.n.01', 'vehicle.n.01', 'herb.n.01',
                    'self-propelled_vehicle.n.01', 'amphibian.n.01', 'canine.n.02', 'domestic_animal.n.01','electronic_equipment.n.01',
                    'device.n.01', 'container.n.01', 'covering.n.02', 'conveyance.n.03', 'commodity.n.01','monkey.n.01',
                    'abstraction.n.06', 'consumer_goods.n.01', 'structure.n.01', 'invertebrate.n.01', 'artifact.n.01', 'ruminant.n.01',
                    'invertebrate.n.01', 'matter.n.03', 'wheeled_vehicle.n.01', 'arthropod.n.01', 'causal_agent.n.01',
                    'reptile.n.01', 'equipment.n.01', 'implement.n.01', 'even-toed_ungulate.n.01', 'garment.n.01','game.n.01',
                    'diapsid.n.01', 'primate.n.02', 'protective_covering.n.01', 'relation.n.01', 'restraint.n.06','ape.n.01',
                    'natural_object.n.01', 'psychological_feature.n.01', 'geological_formation.n.01', 'attribute.n.02','starches.n.01',
                    'obstruction.n.01', 'aquatic_vertebrate','old_world_monkey.n.01','process.n.01','barrier.n.01','new_world_monkey.n.01',
                    'substance.n.07', 'communication.n.02', 'establishment.n.04', 'feline.n.01', 'tool.n.01','clothing.n.01',
                    'food.n.01', 'solid.n.01', 'piece_of_cloth.n.01','brass.n.01','screen.n.05','shelter.n.01','grouse',
                    'machine.n.01', 'vessel.n.03', 'vessel.n.02', 'craft.n.02', 'arachnid.n.01', 'fabric.n.01',
                    'durables.n.01', 'thing.n.12','place_of_business.n.01', 'reproductive_structure.n.01','plant.n.02',
                    'event.n.01', 'material.n.01', 'fastener.n.02', 'woody_plant.n.01', 'measure.n.02','home_appliance.n.01',
                    'mechanism.n.05', 'seafood.n.01', 'cognition.n.01', 'part.n.02', 'organ.n.01', 'group.n.01','game_equipment.n.01',
                    'shape.n.02', 'rodent.n.01', 'military_vehicle.n.01', 'area.n.05', 'mechanical_device.n.01','substance.n.01','nutriment.n.01',
                    'amphibian.n.03', 'salamander.n.01', 'support.n.10', 'produce.n.01', 'natural_elevation.n.01',
                    'mollusk.n.01', 'crustacean.n.01', 'aquatic_mammal.n.01', 'signal.n.01', 'indefinite_quantity.n.01',
                    'act.n.02', 'public_transport.n.01', 'hand_tool.n.01', 'medium.n.01', 'box.n.01', 'state.n.02','kitchen_appliance.n.01',
                    'edible_fruit.n.01', 'toiletry.n.01', 'shellfish.n.01', 'ware.n.01', 'utensil.n.01', 'fur.n.01',
                    'foodstuff.n.02', 'cloak.n.01', 'big_cat.n.01', 'footwear.n.01',  'instrument.n.01',
                     'measuring_instrument.n.01', "sports_equipment.n.01", "stick.n.01", "worker.n.01",
                    "insect.n.01", "computer.n.01", "lepidopterous_insect.n.01", "vine.n.01",
                    "decapod_crustacean.n.01", "musteline_mammal.n.01", "gastropod.n.01",
                    "source_of_illumination.n.01", "edge_tool.n.01", "system.n.01",
                    "drug.n.01", "liquid.n.01", "vegetable.n.01", "course.n.01"}
unfamiliar_names = set([n.split('.')[0] for n in unfamiliar_names])

for n in parent_list:
    if n[0].split('.')[0] in unfamiliar_names:
        existed_names.remove(n[0])

for n in parent_list:
    synset = wn.synset(n[0] )
    if n[0] in existed_names:
        par = get_all_hypernyms(synset)
        for p in par:
            if p.name() in existed_names:
                existed_names.remove(n[0])
                break

pr = [n for n in parent_list if n[0] in existed_names]
print(len(pr))
# print([n.split('.')[0] for n in existed_names])
#
# print([(n[0], [imagenet_mapping[i].name() for i in n[1]]) for n in pr])

for n in pr:
    print(n[0].split('.')[0], [imagenet_mapping[i].name() for i in n[1]], n[1])
# print n[0].split('.')[0], n[1] to evasion_labels.txt
with open('evasion_labels.txt', 'w') as f:
    for n in pr:
        f.write(f"'{n[0].split('.')[0]}',{n[1]}" + '\n')