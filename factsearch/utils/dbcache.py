from collections import defaultdict, OrderedDict
import datetime
import dateutil.parser
import json
import numpy as np
import sqlite3
import subprocess
from tqdm import tqdm
import unicodedata as ud
import sys

import logging
logger = logging.getLogger(__name__)

from aic_nlp_utils.fever import fever_detokenize

class DBCache(object):
    """Sqlite or JSONL backed document storage.

    Implements get_doc_text(doc_id) as in memory db with hash indices.
    
    Based on DRQA DocDB class.
    """

    def __init__(self, db_path, excludekw="", default_date: str="", uni_normalize="NFD"):
        self.path = db_path
        self.excludekw = [e.lower() for e in excludekw.split(";")]
        self.default_date = default_date
        self.uni_normalize = uni_normalize
        if uni_normalize:
            logger.warning("Originally NFD was set by default, I think it was wrong. Check it! It is used by 'get_block_text()' only!")
        if self.path.suffix == ".db":
            self.__import_sqlite()
        elif self.path.suffix == ".jsonl":
            self.__import_jsonl()
        else:
            raise ValueError(f"unknown DB file type: {self.path}, .db and .jsonl supported only")
        # self._clean_titles()
 
    def __import_sqlite(self):
       with sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES) as connection:
            logger.info(f"reading SQLite database to RAM")
            logger.info(f"excluding keywords: {self.excludekw}")

            def hasexcludedkw(kws):
                for k in kws:
                    for e in self.excludekw:
                        if e.startswith(k):
                            return True
                return False

            nrows = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            logger.info(f"processing total {nrows} rows")

            cursor = connection.cursor()
            cursor.execute("SELECT id, did, bid, date, keywords, text FROM documents")
            self.txts = []
            self.kws = []
            self.dates = []
            self.id2row = {}
            self.ids = []
            self.dids = []
            self.bids = []
            self.titles = set()
            self.bodies = set()
            self.did2blocks = defaultdict(list) # maps dids to row numbers

            row = 0
            nexcluded = 0
            for id_, did, bid, date_, keywords, text in tqdm(cursor.fetchall(), total=nrows, mininterval=10.0):
                keywords = [k.lower() for k in keywords.split(";")]
                if hasexcludedkw(keywords):
                    nexcluded += 1
                    continue
                self.txts.append(text)
                self.kws.append(keywords)
                if isinstance(date_, str):
                    # date_ = datetime.datetime.fromisoformat(date_) # should wortk in Python 3.11+
                    date_ = dateutil.parser.isoparse(date_)
                else:
                    date_ = datetime.datetime.utcfromtimestamp(date_ * 1e-9) 
                self.dates.append(date_)
                self.id2row[id_] = row
                self.ids.append(id_)
                self.dids.append(did)
                self.bids.append(bid)
                if bid == 0:
                    self.titles.add(row)
                else:
                    self.bodies.add(row)
                self.did2blocks[did].append(row) # expects that the blocks are sorted in DBs
                row += 1
            cursor.close()
            logger.info(f"blocks imported: {row}, excluded based on keywords: {nexcluded}")                      


    def __import_jsonl(self):
        logger.info(f"reading JSONL database to RAM")
        logger.info(f"excluding keywords: {self.excludekw}")

        def hasexcludedkw(kws):
            for k in kws:
                for e in self.excludekw:
                    if e.startswith(k):
                        return True
            return False

        self.txts = []
        self.kws = []
        self.dates = []
        self.id2row = {}
        self.ids = []
        self.dids = []
        self.bids = []
        self.titles = set()
        self.bodies = set()
        self.did2blocks = defaultdict(list) # maps dids to row numbers

        nrows = int(subprocess.check_output(f"wc -l {self.path}", shell=True).split()[0])
        logger.info(f"processing total {nrows} rows")

        row = 0
        nexcluded = 0
        with open(self.path, 'r') as json_file:
            data = []
            for jline in tqdm(json_file, total=nrows, mininterval=1.0):
                rec = json.loads(jline, object_pairs_hook=OrderedDict)
                id_ = rec['id'] 
                did = rec['did'] 
                bid = rec['bid'] 
                date_ = rec.get('date', self.default_date) 
                keyword_str = rec.get('keywords', '')
                keywords = [k.lower() for k in keyword_str.split(";")] if len(keyword_str) > 0 else []
                text = rec['text']

                if hasexcludedkw(keywords):
                    nexcluded += 1
                    continue

                self.txts.append(text)
                self.kws.append(keywords)
                if isinstance(date_, str):
                    # date_ = datetime.datetime.fromisoformat(date_) # should wortk in Python 3.11+
                    date_ = dateutil.parser.isoparse(date_)
                else:
                    date_ = datetime.datetime.utcfromtimestamp(date_ * 1e-9) 
                self.dates.append(date_)
                self.id2row[id_] = row
                self.ids.append(id_)
                self.dids.append(did)
                self.bids.append(bid)
                if bid == 0:
                    self.titles.add(row)
                else:
                    self.bodies.add(row)
                self.did2blocks[did].append(row) # expects that the blocks are sorted in DBs
                row += 1
            logger.info(f"blocks imported: {row}, excluded based on keywords: {nexcluded}") 


    def get_block_ids(self):
        return self.id2row.keys()

    def get_document_ids(self):
        return self.did2blocks.keys()

    def get_block_text(self, id_):
        if self.uni_normalize:
            return self.txts[self.id2row[ud.normalize(self.uni_normalize, id_)]]
        else:
            return self.txts[self.id2row[id_]]


    def get_block_texts(self, did, f=lambda txt: txt, title=True):
        return OrderedDict((self.ids[i], f(self.txts[i])) for i in self.did2blocks[did] if title or (self.bids[i] != 0))


    def get_document(self, did, f=fever_detokenize, title=True, block_join="\n\n"):
        blocks = [f(self.txts[i]) for i in self.did2blocks[did] if title or (self.bids[i] != 0)]
        if block_join is None:
            return blocks
        return block_join.join(blocks)
    

    def _clean_titles(self):
        # NOT USED NOW!
        # print("_clean_titles ===============================")
        # helper function to clean titles prepended to blocks (an experimental way to add context to paragraphs)
        # TODO this function should be likely generalized 
        for blocks in self.did2blocks.values():
            title = None
            for i in blocks: # find title - is there a better way?
                print(self.bids[i], self.txts[i])
                if self.bids[i] == 0:
                    title = self.txts[i]
            
            # print(f"_clean_titles: title = {title}")
            if title:
                for i in blocks:
                    if self.bids[i] != 0:
                        txt = self.txts[i]
                        if txt.startswith(title):
                            # print(f"_clean_titles: title = {title}")
                            self.txts[i] = txt[len(title):].strip()


    def hasid(self, id_):
        return id_ in self.id2row

    def id2did(self, id_):
        return self.dids[self.id2row[id_]]

    def id2bid(self, id_):
        return self.bids[self.id2row[id_]]

    def id2date(self, id_):
        return self.dates[self.id2row[id_]]

    def did2ids(self, did):
        return [self.ids[i] for i in self.did2blocks[did]]

    def did2date(self, did):
        return self.dates[self.did2blocks[did][0]]
    
    def istitle(self, id_):
        return self.hasid(id_) and self.id2bid(id_) == 0
    
    def __len__(self):
        return len(self.id2row)
    
    def __getitem__(self, id_, detokenize=fever_detokenize):
        row = self.id2row[id_]
        res = {"id": id_, "did": self.dids[row], "bid": self.bids[row], "date": self.dates[row], "keywords": self.kws[row], "text": detokenize(self.txts[row])}
        return res