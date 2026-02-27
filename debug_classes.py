import pickle, numpy as np
with open('models/ui_detector_rf.pkl', 'rb') as f:
    d = pickle.load(f)

classes = d['classes']
model = d['model']

print('Saved class list (10 total):')
for i, c in enumerate(classes):
    print(f'  index {i}: {c}')

print()
print('Model actually trained on (6 classes):')
print('  model.classes_ =', model.classes_)
for i, mc in enumerate(model.classes_):
    print(f'  model output {i} -> class index {mc} -> "{classes[mc]}"')

print()
print('BUG: predict_proba returns 6 columns, but code uses argmax')
print('     and maps to 10-class list, causing WRONG class names!')
print()
print('Example: if model predicts column 5 as highest prob:')
print(f'  Code does:   classes[5] = "{classes[5]}"')
print(f'  Should do:   classes[model.classes_[5]] = "{classes[model.classes_[5]]}"')
