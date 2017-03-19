# -*- coding: utf-8 -*-
"""
Created on  三月 16 15:18 2017

@author: aeloyq
"""
import argparse
import logging
import os
import subprocess
import tarfile
import urllib2
import uuid


OUTPUT_DIR = './data'
PREFIX_DIR = './share/nonbreaking_prefixes'

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", type=str, help="Source language",
                    default="en")
parser.add_argument("-t", "--target", type=str, help="Target language",
                    default="fr")
parser.add_argument("--source-dev", type=str, default="entest.en",
                    help="Source language dev filename")
parser.add_argument("--target-dev", type=str, default="zhtest.zh",
                    help="Target language dev filename")
parser.add_argument("--source-vocab", type=int, default=400000,
                    help="Source language vocabulary size")
parser.add_argument("--target-vocab", type=int, default=6540,
                    help="Target language vocabulary size")


def download_and_write_file(url, file_name):
    logger.info("Downloading [{}]".format(url))
    if not os.path.exists(file_name):
        path = os.path.dirname(file_name)
        if not os.path.exists(path):
            os.makedirs(path)
        u = urllib2.urlopen(url)
        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        logger.info("...saving to: %s Bytes: %s" % (file_name, file_size))
        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % \
                (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print status,
        f.close()
    else:
        logger.info("...file exists [{}]".format(file_name))


def extract_tar_file_to(file_to_extract, extract_into, names_to_look):
    extracted_filenames = []
    try:
        logger.info("Extracting file [{}] into [{}]"
                    .format(file_to_extract, extract_into))
        tar = tarfile.open(file_to_extract, 'r')
        src_trg_files = [ff for ff in tar.getnames()
                         if any([ff.find(nn) > -1 for nn in names_to_look])]
        if not len(src_trg_files):
            raise ValueError("[{}] pair does not exist in the archive!"
                             .format(src_trg_files))
        for item in tar:
            # extract only source-target pair
            if item.name in src_trg_files:
                file_path = os.path.join(extract_into, item.path)
                if not os.path.exists(file_path):
                    logger.info("...extracting [{}] into [{}]"
                                .format(item.name, file_path))
                    tar.extract(item, extract_into)
                else:
                    logger.info("...file exists [{}]".format(file_path))
                extracted_filenames.append(
                    os.path.join(extract_into, item.path))
    except Exception as e:
        logger.error("{}".format(str(e)))
    return extracted_filenames


def tokenize_text_files(files_to_tokenize, tokenizer):
    name=files_to_tokenize
    logger.info("Tokenizing file [{}]".format(name))
    out_file = os.path.join(
        OUTPUT_DIR, os.path.basename(name) + '.tok')
    logger.info("...writing tokenized file [{}]".format(out_file))
    var = ["perl", tokenizer,  "-l", name.split('.')[-1]]
    if not os.path.exists(out_file):
        with open(name, 'r') as inp:
            with open(out_file, 'w', 0) as out:
                subprocess.check_call(
                    var, stdin=inp, stdout=out, shell=False)
    else:
        logger.info("...file exists [{}]".format(out_file))


def create_vocabularies(tr_files, preprocess_file):
    src_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            args.source, args.target, args.source))
    trg_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            args.source, args.target, args.target))
    src_filename = 'en.en.tok'
    trg_filename = 'zh.zh.tok'
    logger.info("Creating source vocabulary [{}]".format(src_vocab_name))
    if not os.path.exists(src_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, src_vocab_name, args.source_vocab,
            os.path.join(OUTPUT_DIR, src_filename)),
            shell=True)
    else:
        logger.info("...file exists [{}]".format(src_vocab_name))

    logger.info("Creating target vocabulary [{}]".format(trg_vocab_name))
    if not os.path.exists(trg_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, trg_vocab_name, args.target_vocab,
            os.path.join(OUTPUT_DIR, trg_filename)),
            shell=True)
    else:
        logger.info("...file exists [{}]".format(trg_vocab_name))
    return src_filename, trg_filename


def merge_parallel(src_filename, trg_filename, merged_filename):
    print "merge_paralleling~"
    with open(src_filename, 'r') as left:
        with open(trg_filename, 'r') as right:
            with open(merged_filename, 'w') as final:
                for lline, rline in equizip(left, right):
                    if (lline != '\n') and (rline != '\n'):
                        final.write(lline[:-1] + ' ||| ' + rline)


def split_parallel(merged_filename, src_filename, trg_filename):
    print "split_paralleling~"
    with open(merged_filename) as combined:
        with open(src_filename, 'w') as left:
            with open(trg_filename, 'w') as right:
                for line in combined:
                    line = line.split('|||')
                    left.write(line[0].strip() + '\n')
                    right.write(line[1].strip() + '\n')


def shuffle_parallel(src_filename, trg_filename):
    logger.info("Shuffling jointly [{}] and [{}]".format(src_filename,
                                                         trg_filename))
    out_src = src_filename + '.shuf'
    out_trg = trg_filename + '.shuf'
    merged_filename = str(uuid.uuid4())
    shuffled_filename = str(uuid.uuid4())
    if not os.path.exists(out_src) or not os.path.exists(out_trg):
        try:
            merge_parallel(src_filename, trg_filename, merged_filename)
            subprocess.check_call(
                "python shuf.py {} > {} ".format(merged_filename, shuffled_filename),
                shell=True)
            split_parallel(shuffled_filename, out_src, out_trg)
            logger.info(
                "...files shuffled [{}] and [{}]".format(out_src, out_trg))
        except Exception as e:
            logger.error("{}".format(str(e)))
    else:
        logger.info("...files exist [{}] and [{}]".format(out_src, out_trg))
    if os.path.exists(merged_filename):
        os.remove(merged_filename)
    if os.path.exists(shuffled_filename):
        os.remove(shuffled_filename)


def main():
    preprocess_file = os.path.join(OUTPUT_DIR, 'preprocess.py')
    tokenizer_file = os.path.join(OUTPUT_DIR, 'tokenizer.perl')
    # Apply tokenizer
    tokenize_text_files('./data/tmp/en.en', tokenizer_file)
    tokenize_text_files('./data/tmp/zh.zh', tokenizer_file)
    tokenize_text_files('./data/tmp/dev/entest.en', tokenizer_file)
    tokenize_text_files('./data/tmp/dev/zhtest.zh', tokenizer_file)
    # Apply preprocessing and construct vocabularies
    src_filename,trg_filename= create_vocabularies('en.en.tok', preprocess_file)
    # Shuffle datasets
    #shuffle_parallel(os.path.join(OUTPUT_DIR, src_filename),
    #                 os.path.join(OUTPUT_DIR, trg_filename))


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('prepare_data')

    args = parser.parse_args()
    main()
