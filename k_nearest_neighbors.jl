using LinearAlgebra
using Statistics
# Size(M^2) District BedRoom BathRoom Floor# Price(in thousand)
# one-hot encoding District -> HuaiKhwang Siam SalaDaeng
X = [
     197 1 0 0 2 2 30;
     35 1 0 0 1 1 19;
     52 1 0 0 2 1 25;
     65 0 1 0 2 2 8;
     55 0 1 0 1 1 25;
     46 0 1 0 1 1 42; 
     76 0 1 0 2 2 42;
     196 0 0 1 3 3 66;
     90 0 0 1 2 2 66;
     96 0 0 1 2 2 23;
     54 0 0 1 1 1 8;
    ]
y = [
     15470;
     4100;
     6990;
     5490;
     7900;
     12380;
     24000;
     35400;
     18100;
     12000;
     7500
    ]

euclidean_distance(v₁, v₂) = sqrt(sum((v₁ - v₂).^2))

function knn(k, n)
  distances = Tuple{Int8, Float64}[]

  for i = 1:size(X)[1]
    xᵢ = reshape(X[i, :], (size(X)[2], 1))
    push!(distances, (i, euclidean_distance(n, xᵢ)))
  end

  neighbor_prices = [ y[index] for (index, _) in sort(distances, lt = (x, y) -> x[2] < y[2])[1:k] ]

  return mean(neighbor_prices)    
end

for k in [1, 3, 5]
  println("\nk = $(k) --> prediction = $(knn(k, transpose([88 1 0 0 2 2 11])))\n")
end