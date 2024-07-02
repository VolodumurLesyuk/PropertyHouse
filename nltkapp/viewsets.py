from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import nltk

from nltk import ne_chunk, pos_tag
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltkapp.serializers import TextSerializer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


class TokenizeTextView(APIView):
    def post(self, request):
        serializer = TextSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            # Виконуємо токенізацію тексту
            tokens = word_tokenize(text)
            return Response({'tokens': tokens}, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class POSTagView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = TextSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            return Response({'result': tagged}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class NERView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = TextSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            tokenized_text = word_tokenize(text)
            tagged_text = pos_tag(tokenized_text)
            chunked_text = ne_chunk(tagged_text)

            entities = []
            current_chunk = []
            for i in chunked_text:
                if type(i) == Tree:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
                    if i.label():
                        entities.append({'entity': " ".join(current_chunk), 'type': i.label()})
                        current_chunk = []
                else:
                    continue

            return Response({'entities': entities}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
