mod tensor;
use ndarray::arr1;
use tensor::Tensor;

fn _test_basic_add_multiply() {
    let a = Tensor::from(arr1(&[2.0, 3.0]).into_dyn());
    let b = Tensor::from(arr1(&[-1.0, 6.0]).into_dyn());
    let c = Tensor::from(arr1(&[4.0, 0.0]).into_dyn());

    let d = &b + &a;
    let e = &d * &c;
    println!("[1.0, 9.0]: {:?}", d.borrow(),);
    println!("[4.0, 0.0]: {:?}", e.borrow(),);
    println!("[2.0, 3.0]: {:?}", a.borrow(),);
    println!("[-1.0, 6.0]: {:?}", b.borrow(),);
    println!("[4.0, 0.0]: {:?}", c.borrow(),);
}

fn _verify_micrograd_logic() {
    let a = Tensor::from(arr1(&[2.0]).into_dyn());
    let b = Tensor::from(arr1(&[-3.0]).into_dyn());
    let c = Tensor::from(arr1(&[10.0]).into_dyn());

    let _d = &a + &b;
    let _e = &a * &b;
    let f = &a + &(&b * &c);
    let g = Tensor::from(arr1(&[4.0]).into_dyn());
    let l = &g * &f;

    println!("{:?}", l)
}

fn _verify_micrograd_backward() {
    let a = Tensor::from(arr1(&[2.0]).into_dyn());
    let b = Tensor::from(arr1(&[0.0]).into_dyn());
    let c = Tensor::from(arr1(&[-3.0]).into_dyn());

    let d = Tensor::from(arr1(&[1.0]).into_dyn());
    let e = Tensor::from(arr1(&[6.8813735870195432]).into_dyn());
    let f = &a * &c;
    let g = &b * &d;
    let h = &f + &g;

    let i = &h + &e;

    let j = i.tanh();

    j.backward();

    println!("{:?}", j)
}

fn _check_operation_double_variable() {
    let a = Tensor::from(arr1(&[3.0]).into_dyn());
    let b = &a + &a;
    b.backward();
    println!("{:?}", b);

    let c = Tensor::from(arr1(&[3.0]).into_dyn());
    let d = &c * &c;
    d.backward();
    println!("{:?}", d);
}

fn main() {
    _check_operation_double_variable();
    // _test_basic_add_multiply();
}
