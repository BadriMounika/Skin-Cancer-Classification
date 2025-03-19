import React, { useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import axios from "axios";
import { addUser } from "./utils/userSlice";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Body from "./pages/Body";
import Contact from "./pages/Contact";
import Upload from "./pages/Upload";
import AboutPage from "./pages/About";
import HomePage from "./pages/Home";
import { toast } from "react-hot-toast";
import Cookies from "js-cookie";
import Dashboard from "./pages/Dashboard";
import Unauthorized from "./pages/Unauthorized";

const BASE_URL = import.meta.env.VITE_API_BASE_URL;

const ProtectedRoute = ({ children, allowedRoles }) => {
  const user = useSelector((state) => state.user);

  if (user.loading) {
    return <div>Loading...</div>; // Or a better loading UI
  }

  const isAuthenticated = Boolean(user?.name && user?.emailId);
  const hasRequiredRole = allowedRoles?.includes(user?.role);

  if (!isAuthenticated) {
    toast.error("Unauthorized access! Please log in.");
    return <Navigate to="/login" replace />;
  }

  if (!hasRequiredRole) {
    toast.error("Access denied! Insufficient permissions.");
    return <Navigate to="/unauthorized" replace />;
  }

  return children;
};


// Separate Wrapper Component to Handle Fetching User
const AppWrapper = () => {
  const dispatch = useDispatch();

  const userData = useSelector((state) => state.user);


  useEffect(() => {
    const fetchUser = async () => {
      if (userData.name && userData.emailId && userData.role) return; // Prevent duplicate API calls
      const token = Cookies.get("token");
      if (!token) return;
      try {
        const res = await axios.get(`${BASE_URL}/api/user/user`, {
          withCredentials: true,
        });

        console.log(res);

        dispatch(addUser({
          name: res.data.fullName,
          emailId: res.data.email,
          role: res.data.role,
          loading: false,
        }));
      } catch (err) {
        console.error("Error fetching user:", err);
        if (err.response?.status === 401) {
          toast.error("Session expired! Please log in again.");
        }
      }
    };

    fetchUser();
  }, [dispatch, userData.name, userData.emailId, userData.role]); // Add dependencies properly


  return <App />;
};

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<Body />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/contact" element={<Contact />} />

        <Route path="/unauthorized" element={<Unauthorized />} />
        <Route path="/dashboard" element={<ProtectedRoute allowedRoles={["admin"]}>
          <Dashboard />
        </ProtectedRoute>} />

        {/* Protected Route */}
        <Route
          path="/upload"
          element={
            <ProtectedRoute allowedRoles={["admin", "user"]}>
              <Upload />
            </ProtectedRoute>
          }
        />
      </Route>
    </Routes >
  );
};

// Wrap App with BrowserRouter
const Root = () => (
  <BrowserRouter>
    <AppWrapper />
  </BrowserRouter>
);

export default Root;
