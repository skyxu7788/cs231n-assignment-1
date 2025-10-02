#!/usr/bin/env python3
import inspect
import unittest
import argparse
from graderUtil import graded, CourseTestRunner, GradedTestCase
from subprocess import Popen
from subprocess import DEVNULL, STDOUT, check_call
import torch
import numpy as np
import pickle
import os

from autograde_utils import if_text_in_py, text_in_cell, assert_allclose

# import student submission
import submission

#########
# TESTS #
#########


class Test_1(GradedTestCase):

    def setUp(self):

        self.test_knn = submission.classifiers.k_nearest_neighbor.KNearestNeighbor()
        self.sol_knn = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.xcs231n.classifiers.k_nearest_neighbor.KNearestNeighbor(),
        )

        np.random.seed(231)

        # train
        self.X_train = np.random.randn(50, 10)
        self.y_train = np.random.randint(5, size=(50))
        self.test_knn.train(self.X_train, self.y_train)
        self.sol_knn.train(self.X_train, self.y_train)

    @graded()
    def test_0(self):
        """1-0-basic: two loops"""

        np.random.seed(231)

        # test
        X_test = np.random.randn(20, 10)
        test_out = self.test_knn.compute_distances_two_loops(X_test)
        sol_out = self.sol_knn.compute_distances_two_loops(X_test)

        self.assertTrue(np.allclose(test_out, sol_out), msg="Student generated distances for two loops knn doesn't match expected value")

    @graded()
    def test_1(self):
        """1-1-basic: no scipy.linalg"""

        knn_path = os.path.join(
            os.path.dirname(__file__),
            "./submission/xcs231n/classifiers/k_nearest_neighbor.py",
        )
        if_function = if_text_in_py(knn_path, "scipy") or if_text_in_py(
            knn_path, "linalg"
        )
        self.assertTrue(
            not if_function, msg="Do not use scipy or linalg anywhere in KNN!"
        )

    @graded(is_hidden=True)
    def test_2(self):
        """1-2-hidden: prediction labels"""

        np.random.seed(231)

        # dists independent of other functions
        dists = np.random.randn(20, 50)
        sol_y_pred = self.sol_knn.predict_labels(dists)
        test_y_pred = self.test_knn.predict_labels(dists)

        self.assertTrue(
            np.allclose(test_y_pred, sol_y_pred),
            msg="Student generated predictions for two loops knn don't match expected values",
        )

    @graded(is_hidden=True)
    def test_3(self):
        """1-3-hidden: no loops"""

        np.random.seed(231)

        # test
        X_test = np.random.randn(20, 10)
        test_out = self.test_knn.compute_distances_no_loops(X_test)
        sol_out = self.sol_knn.compute_distances_no_loops(X_test)

        # TODO: check why nan occurs
        # For now exclude nan
        mask = ~(np.isnan(test_out) | np.isnan(sol_out))
        self.assertTrue(np.allclose(test_out[mask], sol_out[mask]))

    @graded()
    def test_4(self):
        """1-4-basic: no loops vs two loops speedup"""

        out_lst = text_in_cell(
            os.path.join(os.path.dirname(__file__), "./submission/knn.ipynb"),
            "no_loop",
        )
        out_twoloop = out_lst[0]
        out_noloop = out_lst[2]

        time_twoloop = float(out_twoloop.split(" ")[-2])
        time_noloop = float(out_noloop.split(" ")[-2])

        # Correct if the speedup is more than 5x
        speedup = time_twoloop / time_noloop
        self.assertTrue(speedup > 5)

    @graded()
    def test_5(self):
        """1-5-basic: cross validation"""

        cv_output = text_in_cell(
            os.path.join(os.path.dirname(__file__), "./submission/knn.ipynb"),
            "cross_validation",
        )
        acc = float(cv_output[0].split(":")[-1])
        self.assertGreater(acc, 0.28)


class Test_2(GradedTestCase):

    def setUp(self):

        self.W = np.random.randn(50, 10) * 0.0001
        self.X = np.random.randn(100, 50) * 256 - 128  # random images
        self.y = np.random.randint(10, size=(100))  # 100 images

    @graded()
    def test_0(self):
        """2-0-basic: naive"""

        test_loss_no_reg, test_grad_no_reg = submission.classifiers.softmax_loss_naive(self.W, self.X, self.y, 0.0)
        test_loss_reg, test_grad_reg =  submission.classifiers.softmax.softmax_loss_naive(self.W, self.X, self.y, 5e1)
        sol_loss_no_reg, sol_grad_no_reg = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.classifiers.softmax.softmax_loss_naive(self.W, self.X, self.y, 0.0),
        )
        sol_loss_reg, sol_grad_reg = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.classifiers.softmax.softmax_loss_naive(self.W, self.X, self.y, 5e1)
        )

        self.assertTrue(
            np.allclose(test_loss_no_reg, sol_loss_no_reg, atol=1e-2, rtol=0), 
            msg="Student generated loss with no regularization doesn't match the expected value"
        )

        self.assertTrue(
            np.allclose(test_grad_no_reg, sol_grad_no_reg, atol=1e-2, rtol=0), 
            msg="Student generated gradient with no regularization doesn't match the expected value"
        )

        self.assertTrue(
            np.allclose(test_loss_reg, sol_loss_reg, atol=1e-2, rtol=0),
            msg="Student generated loss with regularization doesn't match the expected value"
        )

        self.assertTrue(
            np.allclose(test_grad_reg, sol_grad_reg, atol=1e-2, rtol=0),
            msg="Student generated gradient with regularization doesn't match the expected value",
        )

    @graded(is_hidden=True)
    def test_1(self):
        """2-1-hidden: vectorized"""

        test_loss_no_reg, test_grad_no_reg = submission.classifiers.softmax_loss_vectorized(self.W, self.X, self.y, 0.0)
        test_loss_reg, test_grad_reg = submission.classifiers.softmax.softmax_loss_vectorized(self.W, self.X, self.y, 5e1)
        sol_loss_no_reg, sol_grad_no_reg = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.classifiers.softmax.softmax_loss_vectorized(
                self.W, self.X, self.y, 0.0
            ),
        )
        sol_loss_reg, sol_grad_reg = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.classifiers.softmax.softmax_loss_vectorized(
                self.W, self.X, self.y, 5e1
            ),
        )

        self.assertTrue(
            np.allclose(test_loss_no_reg, sol_loss_no_reg, atol=1e-2, rtol=0),
            msg="Student generated loss with no regularization doesn't match the expected value",
        )

        self.assertTrue(
            np.allclose(test_grad_no_reg, sol_grad_no_reg, atol=1e-2, rtol=0),
            msg="Student generated gradient with no regularization doesn't match the expected value",
        )

        self.assertTrue(
            np.allclose(test_loss_reg, sol_loss_reg, atol=1e-2, rtol=0),
            msg="Student generated loss with regularization doesn't match the expected value",
        )

        self.assertTrue(
            np.allclose(test_grad_reg, sol_grad_reg, atol=1e-2, rtol=0),
            msg="Student generated gradient with regularization doesn't match the expected value",
        )

    @graded()
    def test_2(self):
        """2-2-basic: softmax validation accuracy"""

        validation_output = text_in_cell(
            os.path.join(os.path.dirname(__file__), "./submission/softmax.ipynb"),
            "validate",
        )  # all the output text, a list
        validation_acc = float(validation_output[-1].
                               split(" ")[-1])
        return self.assertGreaterEqual(validation_acc, 0.31) and self.assertLess(
            validation_acc, 0.45, msg="Softmax validation accuracy not within the expected range"
        )

    @graded()
    def test_3(self):
        """2-3-basic: best softmax validation accuracy"""

        validation_output = text_in_cell(
            os.path.join(os.path.dirname(__file__), "./submission/softmax.ipynb"),
            "tuning",
        )  # all the output text, a list
        validation_acc = float(validation_output[-1].split(" ")[-1])
        return self.assertGreaterEqual(validation_acc, 0.35, msg="Best softmax validation accuracy not above the expected value")

    @graded()
    def test_4(self):
        """2-4-basic: best softmax test accuracy"""

        test_output = text_in_cell(
            os.path.join(os.path.dirname(__file__), "./submission/softmax.ipynb"),
            "test",
        )  # all the output text, a list
        validation_acc = float(test_output[-1].split(" ")[-1])
        return self.assertGreaterEqual(validation_acc, 0.35, msg="Best softmax test accuracy not above the expected value")


class Test_3(GradedTestCase):

    def setUp(self):

        # Two Layer Network

        np.random.seed(231)
        N, D, H, C = 3, 5, 50, 7
        X = np.random.randn(N, D)
        y = np.random.randint(C, size=N)
        self.std = 1e-3

        self.model_sub = submission.fc_net.TwoLayerNet(
            input_dim=D, hidden_dim=H, num_classes=C, weight_scale=self.std
        )

        self.model_sol = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.fc_net.TwoLayerNet(
                input_dim=D, hidden_dim=H, num_classes=C, weight_scale=self.std
            ),
        )

        model_sol_params = self.model_sol.params
        self.model_sol.params = self.model_sub.params

        self.scores_sub = self.model_sub.loss(X)
        self.loss_sub, self.grads_sub = self.model_sub.loss(X, y)

        self.scores_sol = self.model_sol.loss(X)
        self.loss_sol, self.grads_sol = self.model_sol.loss(X, y)

        self.model_sol.params = model_sol_params

        # Accuracy

        def get_accuracy(text):
            # "Validation accuracy: 0.4"
            first_line = text[0]
            # "0.4"
            last_word = first_line.split(" ")[-1]
            # 0.4
            return float(last_word)

        self.val_accuracy = get_accuracy(
            text_in_cell(
                os.path.join(
                    os.path.dirname(__file__),
                    "./submission/two_layer_net.ipynb",
                ),
                "val_accuracy",
            )
        )
        self.test_accuracy = get_accuracy(
            text_in_cell(
                os.path.join(
                    os.path.dirname(__file__),
                    "./submission/two_layer_net.ipynb",
                ),
                "test_accuracy",
            )
        )

    @graded()
    def test_0(self):
        """3-0-basic: affine forward and backward"""

        np.random.seed(231)

        N, D1, D2, D3, M = 100, 2, 3, 4, 5
        x = np.random.randn(N, D1, D2, D3)
        w = np.random.randn(D1 * D2 * D3, M)
        b = np.random.randn(M)
        dout = np.random.randn(N, M)

        out_sol, cache_sol = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.layers.affine_forward(x, w, b)
        )
        out_sub, cache_sub = submission.layers.affine_forward(x, w, b)

        dx_sol, dw_sol, db_sol = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.layers.affine_backward(dout, cache_sol),
        )
        dx_sub, dw_sub, db_sub = submission.layers.affine_backward(dout, cache_sub)

        assert_allclose(out_sol, out_sub)
        assert_allclose(dx_sol, dx_sub)
        assert_allclose(dw_sol, dw_sub)
        assert_allclose(db_sol, db_sub)

    @graded()
    def test_1(self):
        """3-1-basic: ReLU forward and backward"""

        np.random.seed(231)

        N, D1, D2 = 100, 2, 3
        x = np.random.randn(N, D1, D2)

        out_sol, cache_sol = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.layers.relu_forward(x)
        )
        out_sub, cache_sub = submission.layers.relu_forward(x)

        dx_sol = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.layers.relu_backward(x, cache_sol)
        )
        dx_sub = submission.layers.relu_backward(x, cache_sub)

        assert_allclose(out_sol, out_sub)
        assert_allclose(dx_sol, dx_sub)

    @graded(is_hidden=True)
    def test_2(self):
        """3-2-hidden: Softmax"""

        np.random.seed(231)

        N, D1 = 100, 10
        x = np.random.randn(N, D1)
        y = np.random.randint(D1, size=(N))  # N images

        out_sol, dx_sol = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.layers.softmax_loss(x, y)
        )
        out_sub, dx_sub = submission.layers.softmax_loss(x, y)

        assert_allclose(out_sol, out_sub)
        assert_allclose(dx_sol, dx_sub)

    @graded(is_hidden=True)
    def test_3(self):
        """3-3-hidden: Init Two Layers Neural Network"""

        sol_keys = set(self.model_sol.params.keys())
        sub_keys = set(self.model_sub.params.keys())
        self.assertTrue(
            sol_keys == sub_keys,
            f"Missing or extra parameters! Expected {sol_keys}, got {sub_keys}.",
        )
        for key in self.model_sol.params.keys():
            # Shape checks: we squeeze to
            sol_param = self.model_sol.params[key]
            sub_param = self.model_sub.params[key]
            self.assertTrue(
                sol_param.squeeze().shape == sub_param.squeeze().shape,
                f"Incorrect shape for {key}. Expected {sol_param.shape}, got"
                " {sub_param.shape} instead.",
            )

    @graded(is_hidden=True)
    def test_4(self):
        """3-4-hidden: Two Layers Neural Network forward"""
        assert_allclose(self.scores_sol, self.scores_sub)

    @graded(is_hidden=True)
    def test_5(self):
        """3-5-hidden: Two Layers Neural Network backward"""
        assert_allclose(self.loss_sol, self.loss_sub)
        self.assertTrue(set(self.grads_sol.keys()) == set(self.grads_sub.keys()))
        for key in self.grads_sol.keys():
            assert_allclose(self.grads_sol[key], self.grads_sub[key])

    @graded()
    def test_6(self):
        """3-6-basic: Validation accuracy threshold"""
        self.assertGreaterEqual(self.val_accuracy, 0.48)

    @graded()
    def test_7(self):
        """3-7-basic: Test accuracy threshold"""
        self.assertGreaterEqual(self.test_accuracy, 0.48)


class Test_4(GradedTestCase):

    def setUp(self):

        self.softmax_test_accuracy = float(
            text_in_cell(
                os.path.join(os.path.dirname(__file__), "./submission/features.ipynb"),
                "svm_test_accuracy",  # this test tag is old from SVM
            )[0]
        )
        self.nn_test_accuracy = float(
            text_in_cell(
                os.path.join(os.path.dirname(__file__), "./submission/features.ipynb"),
                "nn_test_accuracy",
            )[0]
        )

    @graded()
    def test_0(self):
        """4-0-basic: SVM accuracy check"""
        self.assertGreaterEqual(self.softmax_test_accuracy, 0.41)

    @graded()
    def test_1(self):
        """4-1-basic: Neural net accuracy check"""
        self.assertGreaterEqual(self.nn_test_accuracy, 0.56)


class Test_5(GradedTestCase):

    def setUp(self):

        np.random.seed(231)
        N, D, H1, H2, C = 2, 15, 20, 30, 10
        X = np.random.randn(N, D)
        y = np.random.randint(C, size=(N,))
        reg = 1.0

        self.model_sub = submission.fc_net.FullyConnectedNet(
            [H1, H2],
            input_dim=D,
            num_classes=C,
            reg=reg,
            weight_scale=5e-2,
            dtype=np.float64,
        )

        self.model_sol = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.fc_net.FullyConnectedNet(
                [H1, H2],
                input_dim=D,
                num_classes=C,
                reg=reg,
                weight_scale=5e-2,
                dtype=np.float64,
            ),
        )

        model_sol_params = self.model_sol.params
        self.model_sol.params = self.model_sub.params

        self.scores_sub = self.model_sub.loss(X)
        self.loss_sub, self.grads_sub = self.model_sub.loss(X, y)

        self.scores_sol = self.model_sol.loss(X)
        self.loss_sol, self.grads_sol = self.model_sol.loss(X, y)

        self.model_sol.params = model_sol_params

        # Accuracies

        acc_string = text_in_cell(
            os.path.join(
                os.path.dirname(__file__),
                "./submission/FullyConnectedNets.ipynb",
            ),
            "val_test_accuracy",
        )
        self.test_accuracy = float(acc_string[-1].split(" ")[-1])
        self.val_accuracy = float(acc_string[-2].split(" ")[-1])

    @graded()
    def test_0(self):
        """5-0-basic: initialization"""
        sol_keys = set(self.model_sol.params.keys())
        sub_keys = set(self.model_sub.params.keys())
        self.assertTrue(
            sol_keys == sub_keys,
            f"Missing or extra parameters! Expected {sol_keys}, got {sub_keys}.",
        )
        for key in self.model_sol.params.keys():
            # Shape checks: we squeeze to
            sol_param = self.model_sol.params[key]
            sub_param = self.model_sub.params[key]
            self.assertTrue(
                sol_param.squeeze().shape == sub_param.squeeze().shape,
                f"Incorrect shape for {key}. Expected {sol_param.shape}, got"
                f" {sub_param.shape} instead.",
            )

    @graded()
    def test_1(self):
        """5-1-basic: forward"""
        assert_allclose(self.scores_sol, self.scores_sub)

    @graded(is_hidden=True)
    def test_2(self):
        """5-2-hidden: backward"""
        assert_allclose(self.loss_sol, self.loss_sub)
        self.assertTrue(set(self.grads_sol.keys()) == set(self.grads_sub.keys()))
        for key in self.grads_sol.keys():
            assert_allclose(self.grads_sol[key], self.grads_sub[key])

    def gen(self):

        N, D = 4, 5

        np.random.seed(50)
        w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
        dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
        m = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)
        v = np.linspace(0.7, 0.5, num=N * D).reshape(N, D)
        config = {"learning_rate": 1e-2, "m": m, "v": v, "t": 5}
        return w, dw, m, v, config

    @graded(is_hidden=True)
    def test_3(self):
        """5-3-hidden: SGD with momentum"""
        w, dw, m, v, config = self.gen()

        next_w_sol, _ = self.run_with_solution_if_possible(
            submission,
            lambda sub_or_sol: sub_or_sol.sgd_momentum(w, dw, config)
        )
        w, dw, m, v, config = self.gen()
        next_w_sub, _ = submission.sgd_momentum(w, dw, config)
        assert_allclose(next_w_sol, next_w_sub)

    @graded(is_hidden=True)
    def test_4(self):
        """5-4-hidden: RMSProp"""
        w, dw, m, v, config = self.gen()

        next_w_sol, _ = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.rmsprop(w, dw, config)
        )
        w, dw, m, v, config = self.gen()
        next_w_sub, _ = submission.rmsprop(w, dw, config)
        assert_allclose(next_w_sol, next_w_sub)

    @graded(is_hidden=True)
    def test_5(self):
        """5-5-hidden: ADAM"""
        w, dw, m, v, config = self.gen()
        next_w_sol, _ = self.run_with_solution_if_possible(
            submission, lambda sub_or_sol: sub_or_sol.adam(w, dw, config)
        )
        w, dw, m, v, config = self.gen()
        next_w_sub, _ = submission.adam(w, dw, config)
        assert_allclose(next_w_sol, next_w_sub)

    @graded()
    def test_6(self):
        """5-6-basic: validation accuracy"""
        self.assertGreaterEqual(self.val_accuracy, 0.50)

    @graded()
    def test_7(self):
        """5-7-basic: test accuracy"""
        self.assertGreaterEqual(self.test_accuracy, 0.50)


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)

if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
