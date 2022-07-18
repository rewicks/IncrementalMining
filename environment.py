class SimulatedEnvironment:
    def __init__(self, sqlconn, url):
        self.rootURL = url
        self.numAligned = 0
        self.nodes = {} # urlId -> Node
        self.url2urlId = {}
        self.maxLangId = 0

        unvisited = {} # urlId -> Node
        visited = {} # urlId -> Node
        rootURLId = self.Url2UrlId(sqlconn, url)
        self.rootNode = self.CreateNode(sqlconn, visited, unvisited, rootURLId, url)
        self.CreateGraphFromDB(sqlconn, visited, unvisited)
        print("CreateGraphFromDB", len(visited))
        #for node in visited.values():
        #    print(node.Debug())

        self.ImportURLAlign(sqlconn, visited)

        #print("rootNode", rootNode.Debug())
        print("Recombine")
        normURL2Node = {}
        self.Recombine(visited, normURL2Node)
        
        self.rootNode = normURL2Node[self.rootNode.normURL]
        assert(self.rootNode is not None)
        print("rootNode", self.rootNode.Debug())

        self.PruneNodes(self.rootNode)

        self.rootNode.depth = 0
        self.CalcDepth(self.rootNode)

        startNode = Node(sys.maxsize, "START", [], [], None, None)
        startNode.CreateLink("", 0, self.rootNode)
        self.nodes[startNode.urlId] = startNode


        # stop node
        node = Node(0, "STOP", [], [], None, None)
        self.nodes[0] = node

        self.UpdateStats()
        print("nodes", len(self.nodes), 
            "numAligned,", self.numAligned, 
            "maxLangId", self.maxLangId)
        for node in self.nodes.values():
            print(node.Debug())

        print("graph created")

    def CalcDepth(self, node):
        links = []

        # init
        for link in node.links:
            links.append(link)

        while len(links) > 0:
            link = links.pop()
            currDepth = link.parentNode.depth

            if link.childNode.depth > currDepth + 1:
                link.childNode.depth = currDepth + 1

                for childLink in link.childNode.links:
                    links.append(childLink)


    def ImportURLAlign(self, sqlconn, visited):
        #print("visited", visited.keys())
        sql = "SELECT id, url1, url2 FROM url_align"
        val = ()
        sqlconn.mycursor.execute(sql, val)
        ress = sqlconn.mycursor.fetchall()
        assert (ress is not None)

        for res in ress:
            urlId1 = res[1]
            urlId2 = res[2]
            #print("urlId", urlId1, urlId2)

            #print("   ", urlId1, urlId2)
            if urlId1 not in visited or urlId2 not in visited:
                #print("Alignment not in graph", urlId1, urlId2)
                continue

            node1 = visited[urlId1]
            node2 = visited[urlId2]
            node1.alignedNode = node2
            node2.alignedNode = node1

    def UpdateStats(self):
        for node in self.nodes.values():
            if node.alignedNode is not None:
                self.numAligned += 1

            if node.lang > self.maxLangId:
                self.maxLangId = node.lang

            for link in node.links:
                #print(node.Debug(), link.parentNode.Debug())
                assert(node == link.parentNode)
                if link.textLang > self.maxLangId:
                    self.maxLangId = link.textLang

    def PruneNodes(self, rootNode):
        visit = []
        visit.append(rootNode)

        while len(visit) > 0:
            node = visit.pop()
            self.nodes[node.urlId] = node

            # prune links to non-docs
            linksCopy = set(node.links)
            for link in linksCopy:
                childNode = link.childNode
                if len(childNode.docIds) == 0:
                    #print("empty", childNode.Debug())
                    node.links.remove(link)
                elif childNode.urlId not in self.nodes:
                    visit.append(childNode)

    def GetRedirectedNormURL(self, node):
        while node.redirect is not None:
            node =  node.redirect
        normURL = NormalizeURL(node.url)
        return normURL

    def Recombine(self, visited, normURL2Node):
        #print("visited", visited.keys())
        # create winning node for each norm url
        for node in visited.values():
            node.normURL = self.GetRedirectedNormURL(node)
            if node.normURL not in normURL2Node:
                normURL2Node[node.normURL] = node
            else:
                winner = normURL2Node[node.normURL]
                winner.Recombine(node)

        # relink aligned nodes & child nodes to winning nodes
        for node in visited.values():
            if node.alignedNode is not None:
                newAlignedNode = normURL2Node[node.alignedNode.normURL]
                node.alignedNode = newAlignedNode

            for link in node.links:
                childNode = link.childNode
                #print("childNode", childNode.Debug())
                newChildNode = normURL2Node[childNode.normURL]
                #print("newChildNode", newChildNode.Debug())
                #print()
                link.childNode = newChildNode

    def CreateNode(self, sqlconn, visited, unvisited, urlId, url):
        if urlId in visited:
            return visited[urlId]
        elif urlId in unvisited:
            return unvisited[urlId]
        else:
            docIds, langIds, crawlDates, redirectId = self.UrlId2Responses(sqlconn, urlId)
            node = Node(urlId, url, docIds, langIds, crawlDates, redirectId)
            assert(urlId not in visited)
            assert(urlId not in unvisited)
            unvisited[urlId] = node
            return node

    def CreateGraphFromDB(self, sqlconn, visited, unvisited):
        while len(unvisited) > 0:
            (urlId, node) = unvisited.popitem()
            visited[node.urlId] = node
            #print("node", node.Debug())
            assert(node.urlId == urlId)

            if node.redirectId is not None:
                assert(len(node.docIds) == 0)
                redirectURL = self.UrlId2Url(sqlconn, node.redirectId)
                redirectNode = self.CreateNode(sqlconn, visited, unvisited, node.redirectId, redirectURL)
                node.redirect = redirectNode
            else:
                linksStruct = self.DocIds2Links(sqlconn, node.docIds)

                for linkStruct in linksStruct:
                    childURLId = linkStruct[0]
                    childUrl = self.UrlId2Url(sqlconn, childURLId)
                    childNode = self.CreateNode(sqlconn, visited, unvisited, childURLId, childUrl)
                    link = Link(linkStruct[1], linkStruct[2], node, childNode)
                    node.links.add(link)

            #print("   ", node.Debug())
            
    def DocIds2Links(self, sqlconn, docIds):
        docIdsStr = ""
        for docId in docIds:
            docIdsStr += str(docId) + ","

        sql = "SELECT id, url_id, text, text_lang_id FROM link WHERE document_id IN (%s)"
        val = (docIdsStr,)
        sqlconn.mycursor.execute(sql, val)
        ress = sqlconn.mycursor.fetchall()
        assert (ress is not None)

        linksStruct = []
        for res in ress:
            struct = (res[1], res[2], res[3])
            linksStruct.append(struct)

        return linksStruct

    def UrlId2Responses(self, sqlconn, urlId):
        sql = "SELECT id, status_code, crawl_date, to_url_id, lang_id FROM response WHERE url_id = %s"
        val = (urlId,)
        sqlconn.mycursor.execute(sql, val)
        ress = sqlconn.mycursor.fetchall()
        assert (ress is not None)

        docIds = []
        langIds = []
        crawlDates = []
        redirectId = None
        for res in ress:
            if res[1] == 200:
                assert(redirectId == None)
                docIds.append(res[0])
                crawlDates.append(res[2])
                langIds.append(res[4])
            elif res[1] in (301, 302):
                assert(len(docIds) == 0)
                redirectId = res[3]

        return docIds, langIds, crawlDates, redirectId

    def RespId2URL(self, sqlconn, respId):
        sql = "SELECT T1.id, T1.val FROM url T1, response T2 " \
            + "WHERE T1.id = T2.url_id AND T2.id = %s"
        val = (respId,)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchone()
        assert (res is not None)

        return res[0], res[1]


    def UrlId2Url(self, sqlconn, urlId):
        sql = "SELECT val FROM url WHERE id = %s"
        val = (urlId,)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchone()
        assert (res is not None)

        return res[0]

    def Url2UrlId(self, sqlconn, url):
        #print("url",url)
        c = hashlib.md5()
        c.update(url.lower().encode())
        hashURL = c.hexdigest()

        if hashURL in self.url2urlId:
            return self.url2urlId[hashURL]

        sql = "SELECT id FROM url WHERE md5 = %s"
        val = (hashURL,)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchone()
        assert (res is not None)

        return res[0]

    ########################################################################
    def GetNumberAligned(self, path):
        ret = 0
        for transition in path:
            next = transition.nextURLId
            nextNode = self.nodes[next]
            if nextNode.alignedDoc > 0:
                ret += 1
        return ret
