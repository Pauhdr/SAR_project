import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
import sys
import math
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia

        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [
        "all", "title","summary", "section-name",'url'
    ]
    def_field = 'all'
    PAR_MARK = '%'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    all_atribs = ['urls', 'index', 'sindex', 'ptindex', 'docs', 'weight', 'articles',
                  'tokenizer', 'stemmer', 'show_all', 'use_stemming']

    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas

        """
        self.urls = set()  # hash para las urls procesadas,
        self.index = {"all":{},"title":{},"summary":{},"section-name":{},"url":{}}  # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.sindex = {"all":{},"title":{},"summary":{},"section-name":{},"url":{}}  # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {"all":{},"title":{},"summary":{},"section-name":{},"url":{}}  # hash para el indice permuterm.
        self.docs = {}  # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {}  # hash de terminos para el pesado, ranking de resultados.
        self.articles = {}  # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile("\W+")  # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish')  # stemmer en castellano
        self.show_all = False  # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False  # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False  # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()

        # Nuevas variables
        self.docId = 0
        self.artId = 0
            

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################

 
    def increase_docId(self):
        """

        Incrementa el ID del documento para indicar cual va a ser el ID del siguiente documento a procesar.


        """

        self.docId += 1

    def increase_artId(self):
        """

        Incrementa el ID del articulo para indicar cual va a ser el ID del siguiente articulo a procesar.


        """

        self.artId += 1

    def set_showall(self, v: bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v

    def set_snippet(self, v: bool):
        """

        Cambia el modo de mostrar snippet.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v

    def set_stemming(self, v: bool):
        """

        Cambia el modo de stemming por defecto.

        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v

    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################

    def save_info(self, filename: str):
        """
        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename: str):
        """
        Carga la información del índice desde un fichero en formato binario

        """
        # info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article: Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls

    def index_dir(self, root: str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']

        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in files:
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

    ##########################################
    ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
    ##########################################

    def sort_index(self, index: dict) -> dict:
        """
        Ordena un indice invertido por orden alfabetico

        Args:
            index: indice invertido que se debe ordenar

        Returns:
            Dict[str, str]: indice invertido ordenado alfabeticamente

        """

        sorted_inverted_index = {}

        for key in sorted(index.keys()):
            sorted_inverted_index[key] = index[key]

        return sorted_inverted_index

    def parse_article(self, raw_line: str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """

        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(
                subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections')  # no la necesitamos
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article

    def index_file(self, filename: str):
        """

        Indexa el contenido de un fichero.

        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.multifield y self.positional se debe ampliar el indexado


        """

        for position_in_doc, line in enumerate(open(filename)):

            article = self.parse_article(line)
            
            # Comprobamos si hemos procesado ya ese articulo. Si es asi pasamos a la siguiente iteracion
            if (self.already_in_index(article)):
                continue

            else:
                    

                # Para la version minima solo nos importa la seccion "all"
                if(self.multifield):
                    sections = [("all",article["all"]),("title",article["title"]),("summary",article["summary"]),("section-name",article["section-name"]),("url",article["url"])]
            
                else:
                    sections = [("all",article["all"])]

                # Lo primero es tokenizar el articulo j
                for section in sections:
                    # Tokenizamos la seccion de que se encuentra en la segunda posicion de la tupla
                    # La seccion url no se tokeniza de la misma forma que las demas. En esta seccion cada url es un token y por eso 
                    # utilizamos el metodo split()
                    if(section[0] == "url"):
                        tokenize_article = section[1].split()
                    else:
                        tokenize_article = self.tokenize(section[1])

                    # Uniqueamos el articulo     
                    tokenize_article = set(tokenize_article)
                
                    # Ahora vamos a actualizar el indice invertido con las nuevas palabras de este articulo
                    self.update_inverted_index(tokenize_article,self.index[section[0]])

                    # Si se marca que se debe realizar el stemming creamos el indice
                    if (self.stemming):
                        self.make_stemming(tokenize_article,section[0])
                
                    # Si se marca que se debe realizar el permuterm creamos el indice
                    if (self.permuterm):
                        self.make_permuterm(tokenize_article,section[0])

                # Finalmente guardamos el articulo una vez ya procesado y aumentamos el id de articulo
                self.save_article_by_ID(article,position_in_doc)

                

        # Guardamos el documento en su respectivo diccionario
        self.docs[self.docId] = filename
        self.increase_docId()

    def save_article_by_ID(self, article: dict, position_in_doc: int):

        """

        Guarda para cada articulo el documento del que pertenece y su posicion dentro de el. Los articulos
        se diferencian por su docId

        input: "article" es el articulo que se ha extraido del doc y que se esta procesando
                "position_in_doc" la posicion del articulo dentro del documento


        """
        # Añadimos la url para saber que ya se ha procesado
        self.urls.add(article["url"])

        # Guardamos en el diccionario los articulos con su id
        # Para cada articulo guardamos el docId de su documento y la posicion dentro de el
        self.articles[self.artId] = [self.docId, position_in_doc]

        # Aumentamos el indice del siguiente articulo que se procesara
        self.increase_artId()

            

    def update_inverted_index(self, tokenize_article: list, index: dict):

        """
           Para cada articulo que ya tenemos tokenizado, lo recorremos y actualizamos el indice
           invertido con él

           input:   "tokenize_article" es el articulo que se quiere indexar ya tokenizado
                    "index" es el indice invertido que se quiere actualizar
           """

        # Recorremos todos los tokens del articulo
        for token in tokenize_article:
            
            # Si el token no esta en el indice se añade con el docId correspondiente
            if (token not in index):

                index[token] = [1, [self.artId]]

            # Si el token ya esta se actualiza la posting list del token
            else:

                self.update_posting_list(token, index)

    def update_posting_list(self, token: str, index: dict):
        """
            Funcion que dado un token que ya esta indexado, actualiza su posting list con la nueva
            informacion recogida sobre él.

            input:   "token" es el token que se quiere actualizar en el indice
                            "index" es el indice invertido que se quiere actualizar
           """

        # Cogemos la posting list
        posting_list = index[token]

        # Cogemos el numero de documentos en los que aparece el termino
        num_docs = posting_list[0]

        # Cogemos la lista de documentos en la que aparece el termino
        list_of_docs = posting_list[1]

        # El termino no puede estar en la posting list porque lo hemos uniqueado
        # Añadimos el nuevo termino
        list_of_docs.append(self.artId)

        # Actualizamos el numero de documentos en el que aparece cada token
        num_docs = len(list_of_docs)

        posting_list = [num_docs, list_of_docs]

        # Actulizamos la posting list del token
        index[token] = posting_list

    def set_stemming(self, v: bool):
        """

        Cambia el modo de stemming por defecto.

        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v

    def tokenize(self, text: str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()

    def make_stemming(self, tokenize_article: list,section:str):
        """

        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE STEMMING.

        "self.stemmer.stem(token) devuelve el stem del token"

        """
        pass

        stem_article = []

        # Lo primero convertimos los tokens en sus correspondientes stems
        for token in tokenize_article:
            stem_article.append(self.stemmer.stem(token))

        # Actualizamos el indice invertido del stem
        self.update_inverted_index(stem_article, self.sindex[section])


    def make_permuterm(self,tokenize_article: list,section:str):
        """

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE PERMUTERM


        """
        pass
        
        permuted_article = set()
        
        for token in tokenize_article:

            token_permuterm = token+"$"

            for i in range(len(token_permuterm)):
                
                permuterm = token_permuterm[i:] + token_permuterm[:i]

                permuted_article.add(permuterm)

        # Actualizamos el indice invertido del permuterm
        self.update_inverted_index(permuted_article, self.ptindex[section])


    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """
        # Dejamos dos lineas blancas al principio
        print()
        print()

        # Mostramos el numero de archivos indexados
        print("========================================")
        print("Number of indexed files:" +" "+str(len(self.docs)))
        print("----------------------------------------")

        # Mostramos el numero de articulos indexados
        print("Number of indexed articles:"+" "+ str(len(self.articles)))
        print("----------------------------------------")

        # Mostramos el numero de tokens de cada seccion si se ha pedido
        if(self.multifield):

            print("TOKENS:")
            for section in self.fields:
                print("\t# of tokens in "+section+":" +" "+ str(len(self.index[section])))
            print("----------------------------------------")

            # Mostramos las estadisticas de permuterm si se ha pedido
            if(self.permuterm):
                print("PERMUTERMS:")
                for section in self.fields:
                    print("\t# of tokens in "+section+":" +" "+ str(len(self.ptindex[section])))
                print("----------------------------------------")

            # Mostramos las estadisticas de stems si se ha pedido
            if (self.stemming):
                print("STEMS:")
                for section in self.fields:
                    print("\t# of tokens in "+section+":" +" "+ str(len(self.sindex[section])))
                print("----------------------------------------")
            

        else:
            print("TOKENS:")
            print("\t# of tokens in 'all':" + " "+str(len(self.index["all"])))
            print("----------------------------------------")

            # Mostramos las estadisticas de permuterm si se ha pedido
            if(self.permuterm):
                print("PERMUTERMS:")
                print("\t# of tokens in 'all':" +" "+ str(len(self.ptindex["all"])))
                print("----------------------------------------")

            # Mostramos las estadisticas de stems si se ha pedido
            if (self.stemming):
                print("STEMS:")
                print("\t# of tokens in 'all':" +" "+ str(len(self.sindex["all"])))
                print("----------------------------------------")

        # Mostramos si se ha permitido postional queries o no
        if(self.positional):
            print("Positional queries are allowed.")
        else:
            print("Positional queries are NOT allowed.")

        print("========================================")
        
        pass


    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################

    def solve_query(self, query: str, prev: Dict = {}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """
        posibleOperations = ["NOT", "AND", "OR"]
        posting = []
        operator = []
        if query is None or len(query) == 0:
            return []

        for term in query.split(" "):
            if term in posibleOperations:
                operator.append(term)
            else:
                # Comprobamos que el termino no tiene una seccion asociada
                term = term.split(":")
                
                # Si tiene una seccion asociada dividimos
                if(len(term) == 2):
                    section = term[0]
                    term = term[1]

                    post = self.get_posting(term,section)
                   
                # Si no tiene una seccion asociada
                else:
                    term = term[0]
                    post = self.get_posting(term)

                while operator != []:
                    o = operator.pop()
               
                    if o == "NOT":
                       
                        post = self.reverse_posting(post)
                        
                    elif o == "AND":
                        post = self.and_posting(posting, post)
                    elif o == "OR":
                        post = self.or_posting(posting, post)
                posting = post

                

        return posting
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    def get_posting(self, term: str, field: Optional[str] = None):
        """

        Devuelve la posting list asociada a un termino.
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list

        NECESARIO PARA TODAS LAS VERSIONES

        """

        res = []
        field = field if field != None else "all"
        
        term = term.lower()
        if ' ' in term: #Posting list con la ampliación de posicionales
            term = term.split()
            res = self.get_positionals(term, field)

        if '*' in term or '?' in term: #Posting list con la ampliación de permuterms
            res = self.get_permuterm(term, field)

        elif self.use_stemming: #Posting list con la ampliación de stemming
            res = self.get_stemming(term, field)
        
        else: #Posting list sin ampliación
            if term in self.index[field]:
                res = list(self.index[field][term])
       
        return res[1]


    def get_positionals(self, terms: str, index):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        pass
        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################

    def get_stemming(self, term: str, field: Optional[str] = None):
        """

        Devuelve la posting list asociada al stem de un termino.
        NECESARIO PARA LA AMPLIACION DE STEMMING

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        stem = self.stemmer.stem(term)

        if(field == None):
            field = "all"

        return self.sindex[field][stem][1]

    def get_permuterm(self, term: str, field: Optional[str] = None):
        """

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        pass
        
        if(field == None):
            field = "all"

        return self.ptindex[field][term][1]

    def reverse_posting(self, p: list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """

        res = list(self.articles.keys()) #Cogemos todas las noticias
 
        for d in p: #Si está la noticia en la posting list
            if d in res: #Y si aún no la hemos eliminado
                res.remove(d) #Se elimina de la lista del total de noticias

        return res

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    def and_posting(self, p1: list, p2: list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """

        res = [] #Creamos la posting list del resultado
        i, j = 0, 0
        
        while i < len(p1) and j < len(p2): #El pseudocódigo de los apuntes pasado a código
            if p1[i] == p2[j]:
                res.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1
        
        return res
    
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    def or_posting(self, p1: list, p2: list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 o p2

        """

        res = [] #Creamos la posting list del resultado
        i, j = 0, 0

        while i < len(p1) and j < len(p2): #El pseudocódigo de los apuntes pasado a código
            if p1[i] == p2[j]:
                res.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                res.append(p1[i])
                i += 1
            else:
                res.append(p2[j])
                j += 1
        while i < len(p1):
            res.append(p1[i])
            i += 1
        while j < len(p2):
            res.append(p2[j])
            j += 1

        return res
    
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """

        res = [] #Creamos la posting list del resultado
        i, j = 0, 0

        while i < len(p1) and j < len(p2): #El pseudocódigo de los apuntes pasado a código
            if p1[i] == p2[j]:
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                res.append(p1[i])
                i += 1
            else:
                j += 1
        while i < len(p1):
            res.append(p1[i])
            i += 1

        return res
    
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################

    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql: List[str], verbose: bool = True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results

    def solve_and_test(self, ql: List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                # print(line.split('\t'))
                query, ref = line.split('\t')
                reference = int(ref)
                result = len(self.solve_query(query))
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(query)
        return not errors

    def solve_and_show(self, query: str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        result = self.solve_query(query)
        
        print("========================================")
       
        # Si se se pide mostrar todos los resultados
        if(self.show_all):
            total = len(result)

        # Si no, se muestran 10 resultados (SHOW_MAX) siempre y cuando haya 10 o mas resultados
        else:

            if(len(result) >= self.SHOW_MAX):
                total = self.SHOW_MAX

            else:
                total = len(result)

        for id in range(total):
            
            art = self.articles[result[id]]
            doc = art[0]
            pos = art[1]
            file = self.docs[doc]
            lines = open(file).readlines()
            articulo = self.parse_article(lines[pos])
            if self.show_snippet:
                self.get_snippet(id+1, result[id], articulo)
            else:
                print(f"# {id+1} ( {result[id]}) {articulo['title']}:\t{articulo['url']}")

        print("========================================")
        print(f"Number of results: {len(result)}")
        return len(result)
        ################
        ## COMPLETAR  ##
        ################


    def get_snippet(self, count, id, articulo: dict):
        print(f"# {count} ( {id}) {articulo['title']}:\t{articulo['url']}")
        print("Falta el extracto del articulo")









        

