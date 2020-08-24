// finished script. TODO: debug!
// calculate posteriors outside

functions {
real ftft(real x, vector param) {
return exp(x) + param[1]^2 + 2*param[2];
}

}
parameters {
vector[2] p;
}
model {
p ~ normal(0, 1);

integrate_1d(ftft, 0, 1, p);
