import sqlite3

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
                            length integer,
                            sample_rate integer
                        )''')
        except sqlite3.OperationalError as e:
            # assume table already exists? clumsy...
            print(e)
   
    def create_record(self, fname, length, sample_rate):
        c = self.conn.cursor()
        c.execute("insert into samples" +
                  " (fname, length, sample_rate)" +
                  " values (?, ?, ?)",
                    (fname, length, sample_rate))
        self.conn.commit()
