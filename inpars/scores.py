import json
import argparse
import numpy as np
from tqdm import tqdm
from .rerank import Reranker
from .dataset import load_corpus
import ir_datasets
from collections import defaultdict

# def read_synthetic_data(args):
#     rows = []
#     with open(args.input, 'r') as fin:
#         for line in tqdm(fin):
#             row = json.loads(line.strip())
#             if len(row['log_probs']) < args.min_tokens:
#                 continue
#             if len(row['log_probs']) > args.max_tokens:
#                 continue
#             if args.skip_questions_copied_from_context:
#                 if row['question'].lower() in row['doc_text'].lower():
#                     continue
#             rows.append(row)
#     return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input", type=str, required=True,
    #                     help="Input jsonl file with the synthetic queries to be filtered.")
    # parser.add_argument("--dataset", default=None, type=str,
    #                     help="Dataset name from BEIR collection.")
    # parser.add_argument('--dataset_source', default='ir_datasets',
    #                     help="The dataset source: ir_datasets or pyserini")
    # parser.add_argument("--filter_strategy", type=str, required=True,
    #                     help="Filtering strategy: scores or reranker.")
    # parser.add_argument('--keep_top_k', type=int, default=10_000,
    #                     help='Write only top_k best scored query-doc pairs.')
    # parser.add_argument("--output", type=str, required=True,
    #                     help="Path to save the filtered queries.")
    parser.add_argument("--model_name_or_path", type=str,
                        default='castorini/monot5-3b-msmarco-10k', required=False,
                        help="Reranker model to be used in case of filtering_strategy=reranker.")
    # parser.add_argument('--min_tokens', type=int, default=3,
    #                     help='Skip question that have fewer than this number of words.')
    # parser.add_argument('--max_tokens', type=int, default=1000,
    #                     help='Skip question that have more than this number of words.')
    # parser.add_argument('--skip_questions_copied_from_context', action='store_true',
    #                     help='If passed, skip questions that were copied from the passage.')
    parser.add_argument("--device", default=None, type=str,
                        help="CPU or CUDA device.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use FP16 weights during inference.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for inference.")

    args = parser.parse_args()
    # assert args.filter_strategy in ['scores', 'reranker']

    # dataset = read_synthetic_data(args)

    # if args.filter_strategy == "scores":
    #     for line in tqdm(dataset):
    #         line['score'] = np.mean(line['log_probs'])
    # else:
    # corpus = load_corpus(args.dataset, source=args.dataset_source)
    # corpus = dict(zip(corpus['doc_id'], corpus['text']))
    query_path = "../lsr-entities/data/robust04/inparsv2/topics-robust04.tsv"
    queries = {}
    with open(query_path, "r") as f:
        for line in tqdm(f, desc="Reading queries"):
            qid, qtext = line.strip().split("\t")
            queries[qid] = qtext
    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
    docs = {}
    for doc in tqdm(dataset.docs_iter(), desc="Reading documents"):
        docs[doc.doc_id] = doc.title + " " + doc.body.replace("\n", " ")
    triplet_path = "../lsr-entities/data/robust04/inparsv2/triplet_ids.tsv"
    q2doc = defaultdict(set())
    with open(triplet_path, "r") as f:
        for line in tqdm(f, desc="Reading triplets"):
            qid, pos_id, neg_id = line.strip().split("\t")
            q2doc[qid].add(pos_id)
            q2doc[qid].add(neg_id)
    pairs = []
    pair_ids = []
    for qid in q2doc:
        for doc_id in q2doc[qid]:
            pairs.append((queries[qid], docs[doc_id]))
            pair_ids.append((qid, doc_id))
    model = Reranker.from_pretrained(
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        fp16=args.fp16,
        device=args.device,
    )
    pair_scores = model.rescore(pairs)

    score_dict = defaultdict(dict)
    for pair, score in zip(pair_ids, pair_scores):
        score_dict[pair[0]][pair[1]] = score
    json.dumps(score_dict, open(
        "../lsr-entities/data/robust04/inparsv2/monot5_score.json", "w"))
    # for idx, synt_item in enumerate(dataset):
    # synt_item['score'] = query_scores[idx]

    # dataset.sort(key=lambda dataset: dataset['score'], reverse=True)
    # with open(args.output, 'w') as fout:
    # for row in dataset[:args.keep_top_k]:
    # fout.write(json.dumps(row) + '\n')
    # print("Done!")
