import sqlite3
import tqdm

class SampleDB(object):
    def __init__(self, db_file='sample.db', check_same_thread=True):
        self.conn = sqlite3.connect(db_file, check_same_thread=check_same_thread)
    
    def create_if_required(self):
        # called once to create db
        c = self.conn.cursor()
        try:
            c.execute('''create table samples (
                            id integer primary key autoincrement, 
                            fname string,
                            length real,
                            sample_rate integer,
                            peak_db real
                        )''')
            c.execute('''create table clips (
                            id integer primary key autoincrement, 
                            start integer,
                            end integer,
                            peak_db real
                        )''')                        
        except sqlite3.OperationalError as e:
            # assume table already exists? clumsy...
            print(e)
   
    def create_sample_record(self, fname, length, sample_rate):
        c = self.conn.cursor()
        c.execute("insert into samples" +
                  " (fname, length, sample_rate)" +
                  " values (?, ?, ?)",
                    (fname, float(length), int(sample_rate)))
        self.conn.commit()

    def create_sample_records(self, fnames, lengths, sample_rates):
        c = self.conn.cursor()
        for f, l, sr in tqdm.tqdm(zip(fnames, lengths, sample_rates)):            
            c.execute("insert into samples" +
                    " (fname, length, sample_rate)" +
                    " values (?, ?, ?)", (f, float(l), int(sr)))
        self.conn.commit()

    def samples_between_lengths(self, min_len, max_len):
        c = self.conn.cursor()
        c.execute("select id, fname from samples where length >= ? and length <= ?", 
                    (float(min_len), float(max_len)))
        return c.fetchall()     

    def create_clip_records(self, start_end_peak_dbs):
        c = self.conn.cursor()
        for s, e, p in start_end_peak_dbs:
            c.execute("insert into clips" +
                    " (start, end, peak_db)" +
                    " values (?, ?, ?)", (int(s), int(e), float(p)))
        self.conn.commit()

    def set_sample_peak_dbs(self, sample_ids, peak_dbs):
        c = self.conn.cursor()
        for sample_id, peak_db in tqdm.tqdm(zip(sample_ids, peak_dbs)):
            c.execute("update samples set peak_db=? where id=?", 
                        (float(peak_db), sample_id))
        self.conn.commit()

