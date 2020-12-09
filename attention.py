attn = model.t_encoder.layers[0].self_attn

weights = []
for m in attn.modules():
    print('Module name :', m)
    for nm, p in m.named_parameters(recurse=False):
        print(nm)
        weights.append(p)

        
w = weights[0]
print(w.shape)

dim = int(w.shape[0] / 3)

q = w[:dim]
k = w[dim : 2 * dim]
v = w[2 * dim :]


out = F.softmax(torch.mul( (torch.mul(q, torch.transpose(k,0,1))/8) , v))


plt.imshow(out.detach().numpy())
