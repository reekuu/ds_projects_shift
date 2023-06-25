import psycopg2


class Connector:

    def __init__(self):
        self.params = dict(dbname="home_credit",
                           user="reekuu",
                           password="reekuu",
                           host='127.0.0.1',
                           port=5432)

    def send(self, query):
        try:
            # connect to the PostgreSQL server & create a cursor
            print('Connecting to the PostgreSQL database...')
            conn = psycopg2.connect(**self.params)
            cur = conn.cursor()

            # execute a query & close the communication
            print('Executing query...')
            cur.execute(query)
            conn.commit()
            cur.close()            
            
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            
        finally:
            if conn:
                conn.close()
                print('Database connection closed.')
                
    def read(self, query):
        conn = psycopg2.connect(**self.params)
        df = pd.read_sql(query, conn)
        conn.close()
        return df
