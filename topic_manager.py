from collections import defaultdict

class TopicManager:
    """
    Gerencia a lista de tópicos gerados para garantir consistência 
    e preparar os dados para a consolidação.
    """
    def __init__(self, analisador_llm):
        # Armazena apenas os rótulos únicos: {"Tópico A", "Tópico B", ...}
        self.topicos_unicos = set() 
        # Armazena o mapeamento temporário para o LLM de consolidação (rótulo: contagem/índice)
        self.topicos_com_indice = {} 
        self.analisador_llm = analisador_llm
        self.next_level_indicator = 1 # Para usar no formato [1]

    def add_topic(self, topico_label: str):
        """Adiciona um tópico se ele for novo e não for 'None' ou 'ERRO'."""
        if topico_label and topico_label not in ["None", "ERRO"] and topico_label not in self.topicos_unicos:
            self.topicos_unicos.add(topico_label)
            # Para a consolidação, precisamos de um índice e uma descrição fake
            self.topicos_com_indice[topico_label] = self.next_level_indicator
            self.next_level_indicator += 1

    def get_topicos_str_para_geracao(self) -> str:
        """Retorna os tópicos em formato de string simples para o prompt da Etapa 1."""
        return "\n".join(sorted(list(self.topicos_unicos)))

    def get_topicos_str_para_consolidacao(self) -> str:
        """
        Retorna os tópicos no formato complexo exigido pelo prompt de consolidação:
        [1] Tópico A: [Tópico A (sem descrição)]
        """
        formatado = []
        for topico, idx in sorted(self.topicos_com_indice.items(), key=lambda item: item[1]):
            # Usamos o próprio tópico como uma descrição fake para atender à sintaxe do prompt de fusão.
            formatado.append(f"[{idx}] {topico}: {topico}")
        return "\n".join(formatado)

    def consolidar_topicos(self) -> dict:
        """
        Executa a consolidação chamando o LLM e atualiza a lista interna de tópicos.
        Retorna o mapeamento de tópicos antigos para novos após a consolidação.
        """
        if not self.topicos_unicos:
            return {}

        topicos_para_prompt = self.get_topicos_str_para_consolidacao()
        
        # Chama o LLM
        modificacao_str = self.analisador_llm.consolidar_topicos(topicos_para_prompt)
        
        # Se LLM não retornar modificações
        if modificacao_str == "None" or not modificacao_str:
            return {topico: topico for topico in self.topicos_unicos} # Retorna mapeamento 1:1

        # 1. Parsear a saída do LLM (modificacao_str pode ter várias linhas)
        # Ex: [1] Technology: Discuss technology... ([1] Digital Literacy, [1] Telecommunications)
        # Assumindo que o LLM retorna apenas a(s) linha(s) de modificação
        
        # Mapeamento do tópico antigo (rótulo) para o novo rótulo consolidado
        mapeamento_consolidado = {topico: topico for topico in self.topicos_unicos}
        
        # Expressão regular para extrair o novo tópico e os tópicos antigos fundidos.
        import re
        regex = r"^\[\d+\]\s*([^:]+):.*?\((.*)\)$"
        
        for linha in modificacao_str.split('\n'):
            match = re.search(regex, linha.strip())
            if match:
                novo_topico = match.group(1).strip()
                topicos_antigos_str = match.group(2)
                
                # Encontra os rótulos antigos (ex: 'Digital Literacy', 'Telecommunications')
                # Assumindo que os rótulos antigos vêm no formato [1] Tópico
                topicos_a_fundir = [
                    re.search(r"\]\s*([^,)]+)", t.strip()).group(1).strip()
                    for t in topicos_antigos_str.split(',') if re.search(r"\]\s*([^,)]+)", t.strip())
                ]
                
                # 2. Aplicar a Fusão no Mapeamento
                for antigo in topicos_a_fundir:
                    if antigo in self.topicos_unicos:
                        mapeamento_consolidado[antigo] = novo_topico
                        self.topicos_unicos.discard(antigo) # Remove os tópicos antigos
                
                # 3. Adicionar o novo tópico se não existir
                self.topicos_unicos.add(novo_topico)
        
        # 4. Reconstruir o dicionário interno (self.topicos_com_indice)
        self.topicos_com_indice = {}
        self.next_level_indicator = 1
        for topico in sorted(list(self.topicos_unicos)):
             self.topicos_com_indice[topico] = self.next_level_indicator
             self.next_level_indicator += 1

        return mapeamento_consolidado