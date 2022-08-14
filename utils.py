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
