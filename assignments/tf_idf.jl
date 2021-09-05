using Printf
using Pandas

sentence_1 = "he really really loves coffee I think"
sentence_2 = "my sister drinks coffee every day"
sentence_3 = "my sister loves tea more than coffee"

function term_frequency(document)
  tf = Dict{String, Rational{Int8}}()
  document = split(document, ' ')
  for word in document
    if haskey(tf, word)
      continue
    end
    count = 0
    for word_2 in document
      if word == word_2
        count += 1
      end
    end
    tf[word] = count//length(document)
  end

  tf
end

function document_frequency(documents...)
  distinct_words = Set{String}([word for document in documents for word in split(document, ' ')])
  df = Dict{String, Rational{Int8}}()
  for distinct_word in distinct_words
    count = 0
    for document in documents
      if distinct_word in split(document, ' ')
        count += 1
      end
    end
    
    df[distinct_word] = count//length(documents)
  end

  df
end

term_frequency_inverse_document_frequency(tf, df) = tf * log(1/df)

function pretty_print(d)
  for (i, ele) in enumerate(d)
    if typeof(d) <: Dict
        (k, v) = ele
        if typeof(v) <: Rational
          print("$(k) = $(numerator(v))/$(denominator(v))")
        else
          print("$(k) = $(v)")
        end
    elseif typeof(d) <: Vector{Tuple{String, Float64}}
        print("$(ele[1]) = ")
        @printf("%.4f", ele[2])
    end

    if i != length(d)
      print(", ")
    end
    if i%5 == 0
      println()
    end
  end
  println()
end

printstyled("\nI) Show step by step to calculate TF-ITF score for each word in each sentence.\n\n", color = :yellow)

printstyled("Step 1 : count the occurrences of each word\n", color = :green)
tfs = []
for (i, sentence) in enumerate([sentence_1, sentence_2, sentence_3])
  tf = term_frequency(sentence)
  push!(tfs, tf)
  printstyled("\nDocument $(i): \n", color = :cyan)
  pretty_print(tf)
end

printstyled("\nStep 2 : calculate the document frequencies of each word\n\n", color = :green)
df = document_frequency(sentence_1, sentence_2, sentence_3)
pretty_print(df)


printstyled("\nStep 3 : calculate tf-idf\n\n", color = :green)
tf_idfs = []
for (i, sentence) in enumerate(tfs)
  printstyled("Sentence $(i) : \n", color = :cyan)
  tf_idf = [(k, term_frequency_inverse_document_frequency(v, df[k])) for (k, v) in sentence]
  pretty_print(tf_idf)
  println()
  push!(tf_idfs, tf_idf)
end

printstyled("II) Show term-document matrix with TF-IDF score.\n\n", color = :yellow)

pre_data = Dict{String, Vector{Float64}}()

for word in keys(df)
  pre_data[word] = [ 0 for _ in range(1, 3)]
end

for (i, sentence) in enumerate([split(sentence_1, ' '), split(sentence_2, ' '), split(sentence_3, ' ')])
  for word in sentence
    pre_data[word][i] = round(filter(x -> x[1] == word, tf_idfs[i])[1][2], digits=4)
  end
end

dataframe = DataFrame(pre_data, index=["sentence_1", "sentence_2", "sentence_3"])
print(dataframe)

printstyled("\nIII) Show step to retrieve the most similar sentence to this one.\n\t\"My brother neither drinks coffee nor tea.\"\n\n", color = :yellow)
new_sentence = "my brother neither drinks coffee nor tea"
for (i, sentence) in enumerate([split(sentence_1, ' '), split(sentence_2, ' '), split(sentence_3, ' ')])
  score = 0
  printstyled("Sentence $(i)'s similarity score : ", color = :cyan)
  for (k, word) in enumerate(split(new_sentence, ' '))
    if word in sentence
      score += dataframe[word][i]
      if k == 1 
        @printf("%.4f ", dataframe[word][i])
      elseif k > 1 && score > 0
        @printf("+ %.4f ", dataframe[word][i])
      end
    end
  end
  @printf("= %.4f\n", score)
end
printstyled("\nTherefore the most similar sentence to the sentence is sentence_2\n\n", color = :red)