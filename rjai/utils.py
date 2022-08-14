import os
import pickle
import editdistance
import configparser

class MySQL:
    def __init__(self, config_file):
        import mysql.connector
        config = configparser.ConfigParser()
        config.read(config_file)
        self.mydb = mysql.connector.connect(
        host=config["mysql"]["host"],
        user=config["mysql"]["user"],
        passwd=config["mysql"]["password"],
        database=config["mysql"]["database"],
        charset='utf8'
        )
        self.mydb.autocommit = False
        self.mycursor = self.mydb.cursor(buffered=True)

######################################################################################
def GetLanguages(configFile):
    filePath = 'pickled_domains/Languages'
    if not os.path.exists(filePath):
        print("mysql load Languages")
        sqlconn = MySQL(configFile)
        languages = Languages(sqlconn)
        os.makedirs("pickled_domains", exist_ok=True)
        with open(filePath, 'wb') as f:
            pickle.dump(languages, f)
    else:
        print("unpickle Languages")
        with open(filePath, 'rb') as f:
            languages = pickle.load(f)

    return languages

######################################################################################
class Languages:
    def __init__(self, sqlconn):
        self.coll = {}

        sql = "SELECT id, lang FROM language"
        sqlconn.mycursor.execute(sql)
        ress = sqlconn.mycursor.fetchall()
        assert (ress is not None)

        for res in ress:
            self.coll[res[1]] = res[0]
            self.maxLangId = res[0]
        
    def GetLang(self, str):
        str = StrNone(str)
        assert(str in self.coll)
        return self.coll[str]
        # print("GetLang", str)

def StrNone(arg):
    if arg is None:
        return "None"
    else:
        return str(arg)

# Given a candidate url, return the smallest string edit distance.
def FindMinEditDistance(candidate_url, existing_urls):
    distances = [editdistance.eval(candidate_url, curr) for curr in existing_urls]
    min_dist = sys.maxsize 
    min_dist_idx = -1
    for i in range(len(distances)):
        if distances[i] < min_dist:
            min_dist = distances[i]
            min_dist_idx = i
    return min_dist #min_dist_idx
