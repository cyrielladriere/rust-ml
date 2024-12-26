// use std::borrow::Borrow;
// This causes some serious bugs when using .borrow() for interior mutabililty
// because bringing it into scope overwrites correct borrow() function

use ndarray::{arr0, ArrayD};
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use uuid::Uuid;

#[derive(Debug)]
pub struct TensorData {
    pub data: ArrayD<f32>,
    pub grad: Option<ArrayD<f32>>,
    pub _op: Option<String>,
    pub _children: Vec<Tensor>,
    pub _backward: Option<fn(out: &TensorData)>,
    pub _uuid: Uuid,
}

// Wrapper around TensorData, access Tensordata content: tensor.0.borrow()
#[derive(Debug, Clone)]
pub struct Tensor(Rc<RefCell<TensorData>>);

impl TensorData {
    pub fn new(data: ArrayD<f32>) -> TensorData {
        TensorData {
            data,
            grad: None,
            _op: None,
            _children: Vec::new(),
            _backward: None,
            _uuid: Uuid::new_v4(),
        }
    }
}

impl Tensor {
    pub fn new(data: TensorData) -> Tensor {
        Tensor(Rc::new(RefCell::new(data)))
    }

    pub fn tanh(&self) -> Tensor {
        let data = self.borrow().data.clone();
        // Tanh forward
        let tanh_data = data.mapv(|x| x.tanh());

        let mut new_tensor_data = TensorData::new(tanh_data);
        new_tensor_data._op = Some(String::from("tanh"));
        new_tensor_data._children = vec![self.clone()];

        fn backward(out: &TensorData) {
            let tanh_out = out.data.clone();
            let grad = out.grad.clone().unwrap();

            // Tanh derivative: (1 - tanh^2) * grad
            let grad_input = grad * (1.0 - &tanh_out * &tanh_out);
            out._children[0].borrow_mut().grad = Some(grad_input);
        }
        new_tensor_data._backward = Some(backward);

        Tensor::new(new_tensor_data)
    }

    pub fn relu(&self) -> Tensor {
        let data = self.borrow().data.clone();
        // ReLU forward: max(0, x)
        let relu_data = data.mapv(|x| if x > 0.0 { x } else { 0.0 });

        let mut new_tensor_data = TensorData::new(relu_data);
        new_tensor_data._op = Some(String::from("relu"));
        new_tensor_data._children = vec![self.clone()];

        fn backward(out: &TensorData) {
            let relu_out = out.data.clone();
            let grad = out.grad.clone().unwrap();

            // ReLU derivative: 1 if x > 0, 0 otherwise
            let grad_input = grad * relu_out.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            out._children[0].borrow_mut().grad = Some(grad_input);
        }
        new_tensor_data._backward = Some(backward);

        Tensor::new(new_tensor_data)
    }

    pub fn backward(&self) {
        let mut topo: Vec<Tensor> = vec![];
        let mut visited: HashSet<Tensor> = HashSet::new();
        self._build_topo(&mut topo, &mut visited);
        topo.reverse();

        // Should this aray not be just ones with shape of self.data
        self.borrow_mut().grad = Some(arr0(1.0).into_dyn());
        for v in topo {
            // Check if v has a backward function, if so invoke it
            if let Some(backprop) = v.borrow()._backward {
                backprop(&v.borrow());
            }
        }
    }

    fn _build_topo(&self, topo: &mut Vec<Tensor>, visited: &mut HashSet<Tensor>) {
        if visited.insert(self.clone()) {
            self.borrow()._children.iter().for_each(|child| {
                child._build_topo(topo, visited);
            });
            topo.push(self.clone());
        }
    }
}
// Lets us do `tensor.borrow().data` instead of `tensor.0.borrow().data`
impl std::ops::Deref for Tensor {
    type Target = Rc<RefCell<TensorData>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow()._uuid.hash(state);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.borrow()._uuid == other.borrow()._uuid
    }
}

impl Eq for Tensor {}

impl From<ArrayD<f32>> for Tensor {
    fn from(item: ArrayD<f32>) -> Self {
        Tensor::new(TensorData::new(item))
    }
}

impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        let mut new_tensor_data = TensorData::new(&self.borrow().data + &other.borrow().data);
        new_tensor_data._op = Some(String::from("+"));
        // Clone not that expensive because it is a data location/address that we are copying
        new_tensor_data._children = vec![self.clone(), other.clone()];

        fn backward(out: &TensorData) {
            // Derivative of out._children[0]+out._children[1] wrt each is both
            // 1 * out.grad because we want to propagate the gradients from end to beginning
            let grad = out.grad.clone().unwrap();

            // Update gradients of the children
            for child in out._children.iter() {
                // A child with a None for gradient should be set to 0
                let mut child_mut = child.borrow_mut();

                let child_grad = child_mut
                    .grad
                    .clone()
                    .unwrap_or_else(|| arr0(0.0).into_dyn());

                // "&child_grad +" bcs we have to accumulate gradients in case that the same variable is in the equation multiple times
                child_mut.grad = Some(&child_grad + &grad);
            }
        }
        new_tensor_data._backward = Some(backward);

        Tensor::new(new_tensor_data)
    }
}

impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Self::Output {
        let mut new_tensor_data = TensorData::new(&self.borrow().data * &other.borrow().data);
        new_tensor_data._op = Some(String::from("*"));
        new_tensor_data._children = vec![self.clone(), other.clone()];

        fn backward(out: &TensorData) {
            let grad = out.grad.clone().unwrap();

            // Clone data outside the mutable borrow phase to avoid conflicts
            let (left_data, right_data, left_grad, right_grad, children_are_same) = {
                let left_child = out._children[0].borrow();
                let right_child = out._children[1].borrow();

                (
                    left_child.data.clone(),
                    right_child.data.clone(),
                    left_child
                        .grad
                        .clone()
                        .unwrap_or_else(|| arr0(0.0).into_dyn()),
                    right_child
                        .grad
                        .clone()
                        .unwrap_or_else(|| arr0(0.0).into_dyn()),
                    Rc::ptr_eq(&out._children[0], &out._children[1]),
                )
            };

            // If children are the same, mutable borrows of both will cause program to panic
            if children_are_same {
                let mut child_mut = out._children[0].borrow_mut();
                child_mut.grad = Some(&left_grad + &(grad.clone() * (&left_data + &right_data)));
            } else {
                let mut left_child_mut = out._children[0].borrow_mut();
                let mut right_child_mut = out._children[1].borrow_mut();

                left_child_mut.grad = Some(&left_grad + &(grad.clone() * &right_data));
                right_child_mut.grad = Some(&right_grad + &(grad * &left_data));
            }
        }

        new_tensor_data._backward = Some(backward);

        Tensor::new(new_tensor_data)
    }
}
