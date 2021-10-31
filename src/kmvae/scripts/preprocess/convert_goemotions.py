import argparse
import csv
import json


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert GoEmotions dataset.")

    parser.add_argument(
        '--goemotions_input_path',
        default='data/goemotions/data/train.tsv', type=str,
        help='GoEmotions tsv path to load from.')

    parser.add_argument(
        '--goemotions_output_path',
        default='data/goemotions/data/train.json', type=str,
        help='GoEmotions json path to save to.')

    parser.add_argument(
        '--goemotions_emotion_ids_path',
        default='data/goemotions/data/emotions.txt', type=str,
        help='GoEmotions emotion ids path.')

    parser.add_argument(
        '--goemotions_ekman_mapping_path',
        default='data/goemotions/data/ekman_mapping.json', type=str,
        help='GoEmotions Ekman mapping path.')

    return parser


def convert_goemotions(
        goemotions_input_path,
        goemotions_output_path,
        goemotions_emotion_ids_path,
        goemotions_ekman_mapping_path):
    # Read mapping from ids to emotions
    with open(goemotions_emotion_ids_path) as file:
        emotion_ids = file.read().splitlines()
    id_to_emotion = {id: emotion for id, emotion in enumerate(emotion_ids)}

    # Read mapping from ekman style emotions to goemotions emotions
    with open(goemotions_ekman_mapping_path) as file:
        ekman_mapping = json.load(file)
    ekman_mapping['neutral'] = ['neutral']

    # Create mapping from goemotions emotions to ekman style emotions
    ekman_reverse_mapping = {
        emotion: ekman_emotion
        for ekman_emotion, emotions in ekman_mapping.items()
        for emotion in emotions}
    ekman_emotion_to_id = {
        emotion: id for id, emotion in enumerate(ekman_mapping.keys())}

    with open(goemotions_input_path, newline='') as file:
        fieldnames = ('text', 'emotion_ids', 'comment_id')
        reader = csv.DictReader(
            file,
            fieldnames=fieldnames,
            dialect='excel-tab',
            lineterminator='\n')

        sentences = []

        for row in reader:
            text = row['text']

            emotion_ids = row['emotion_ids'].split(',')
            emotion_ids = [int(emotion_id) for emotion_id in emotion_ids]

            emotions = set()
            ekman_emotions = set()
            ekman_emotion_ids = set()
            for emotion_id in emotion_ids:
                emotion = id_to_emotion[emotion_id]
                ekman_emotion = ekman_reverse_mapping[emotion]
                ekman_emotion_id = ekman_emotion_to_id[ekman_emotion]

                emotions.add(emotion)
                ekman_emotions.add(ekman_emotion)
                ekman_emotion_ids.add(ekman_emotion_id)
            emotions = list(emotions)
            ekman_emotions = list(ekman_emotions)
            ekman_emotion_ids = list(ekman_emotion_ids)

            comment_id = row['comment_id']

            sentences.append({
                'text': text,
                'labels': emotions,
                'label_ids': emotion_ids,
                'ekman_emotion': ekman_emotions,
                'ekman_emotion_ids': ekman_emotion_ids,
                'comment_id': comment_id,
            })

    with open(goemotions_output_path, 'w') as file:
        json.dump(sentences, file, indent=4)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    convert_goemotions(**vars(args))
