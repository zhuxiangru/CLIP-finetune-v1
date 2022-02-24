import os
import sys
sys.path.append(os.path.abspath(".") + "/../")
sys.path.append(os.path.abspath("."))
import clip
import torch
import json
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load('ViT-B/32', device)
# 华为云项目
#model, preprocess = clip.load('/home/featurize/work/CLIP/save/CLIP-concept-finetune.ckpt', device, jit=False)
#model, preprocess = clip.load('/home/featurize/work/CLIP/save/checkpoints_new/CLIP-concept-finetune_train_20.ckpt', device, jit=False)
# 对比学习实验
#model, preprocess = clip.load('/home/featurize/work/CLIP/save/contrastive_learning_checkpoints/CLIP-concept-finetune_original_no_clustering.ckpt', device, jit=False)
model, preprocess = clip.load('/home/featurize/work/CLIP/save/contrastive_learning_checkpoints//CLIP-concept-finetune_clustering_modified_classified_0_8.ckpt', device, jit=False)



def load_zh_en_concepts(infilename):
    concept_dict = {}
    with open(infilename, "r", encoding = "utf-8") as infile:
        for line in infile:
            line_list = line.strip("\n").split("\t")
            if len(line_list) >= 2:
                concept_dict[line_list[0]] = line_list[1]
    return concept_dict

def load_concept_image_pair(infilename):
    image_conept_pair = {}
    with open(infilename, "r", encoding = "utf-8") as infile:
        line = infile.readline()
        concept_dict = json.loads(line.strip())
        
    for key, value_list in concept_dict.items():
        for value in value_list:
            if str(value) not in image_conept_pair:
                image_conept_pair[str(value)] = key
    return image_conept_pair

def load_common_concepts(infilename):
    concept_dict = {}
    with open(infilename, "r", encoding = "utf-8") as infile:
        for line in infile:
            line_list = line.strip("\n").split("\t")
            if len(line_list) < 2:
                continue
            if line_list[1] not in concept_dict:
                concept_dict[line_list[1]] = None
    return concept_dict


def main():
    topK = 50
    concepts_dict = load_zh_en_concepts("validates/zh_en_concepts.txt")
    concepts_list = concepts_dict.keys()
    image_concept_pair = load_concept_image_pair("validates/concept_image.json")
    #for image, concept in image_concept_pair.items():
    #    print (image, concept)
    #validate_en_concepts_dict = load_common_concepts("validate_concepts.txt")
    validate_en_concepts_dict = load_common_concepts("validates/zh_en_human_concepts.txt")
    
    pos_label_count = 0
    neg_label_count = 0
        
    for key, value in image_concept_pair.items():
        #print (key, value)
        image_fielpath_jpg = "validates/images/" + key + ".jpg"
        image_fielpath_png = "validates/images/" + key + ".png"
        if os.path.isfile(image_fielpath_jpg):
            image_filepath = image_fielpath_jpg
        elif os.path.isfile(image_fielpath_png):
            image_filepath = image_fielpath_png
        else:
            print ("Path not exist.")
            continue
            
        concept = value
        #print (image_filepath, concept)
        image = Image.open(image_filepath)
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # assure the concept in validate datasets
        if concepts_dict[concept] not in validate_en_concepts_dict:
            print ("concept %s-%s not in validate concept dataset" % (concept, concepts_dict[concept]))
            continue
            
        # Prepare the inputs
        candidate_concept = [concepts_dict[concept]] + [concepts_dict[ele] for ele in concepts_list if ele != concept]
        #print (candidate_concept[0:topK])
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in candidate_concept]).to(device)
            
            
        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(topK)
        #values, indices = similarity[0].topk(50)
        longest_index = 0
        ratio = values[0]/20.0
        gap_flag = False
        for index in range(len(values)):
            if values[index] < ratio:
                longest_index = index
                gap_flag = True
                break
        if gap_flag == False:
            longest_index = -1
        values, indices = values[:longest_index], indices[:longest_index]
        #print (values, indices)

        # Print the result
        #print("\nTop predictions:\n")
        #for value, index in zip(values, indices):
        #    print(f"{candidate_concept[index]:>16s}: {100 * value.item():.2f}%")
        print ("ground: image_path=%s, concept=%s" % (image_filepath, concepts_dict[concept]))
        #print (indices)
        print ("predict: concept=%s" % candidate_concept[indices[0]])
        result_concept_set = set()
        for index in indices:
            result_concept_set.add(candidate_concept[index])
        if concepts_dict[concept] in result_concept_set:
            #print ("yes")
            pos_label_count += 1
        else:
            #print ("no")
            neg_label_count += 1
        print ("pos_label_count=", pos_label_count)
        print ("neg_label_count=", neg_label_count)
        
    total_label_count = pos_label_count + neg_label_count
    acc = pos_label_count / total_label_count
    print ("acc=", acc)

if __name__ == '__main__':
    main()

