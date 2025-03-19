import { Link, NavLink, useNavigate } from "react-router-dom";
import { useSelector, useDispatch } from "react-redux";
import { removeUser } from "../utils/userSlice";
import toast from "react-hot-toast";
import { useState, useEffect } from "react";
import axios from "axios";
import { Menu, X } from "lucide-react";
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
} from "@/components/ui/navigation-menu";

export default function Navbar() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const user = useSelector((state) => state.user);

  const [isAuthenticated, setIsAuthenticated] = useState(
    !!user?.name && !!user?.emailId
  );
  const [isAdminAuthenticated, setIsAdminAuthenticated] = useState(
    user?.role === "admin"
  );
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    setIsAuthenticated(!!user?.name && !!user?.emailId);
    setIsAdminAuthenticated(user?.role === "admin");
  }, [user]);

  const BASE_URL = import.meta.env.VITE_API_BASE_URL;

  const handleLogout = async () => {
    try {
      await axios.post(
        `${BASE_URL}/api/user/logout`,
        {},
        { withCredentials: true }
      );
      dispatch(removeUser());
      toast.success("Logged out successfully!");
      navigate("/login");
    } catch (error) {
      console.error(error);
      toast.error("Error logging out!");
    }
  };

  return isAuthenticated ? (
    <nav className="bg-gradient-to-r from-gray-700 to-gray-900 text-white py-4 px-6 shadow-lg">
      <div className="max-w-8xl mx-auto flex items-center justify-between">
        <div className="text-2xl font-bold">Skin Cancer Classification</div>

        {/* Desktop Navigation */}
        <NavigationMenu>
          <NavigationMenuList className="hidden md:flex space-x-1">
            <NavigationMenuItem>
              <NavLink
                to="/"
                className={({ isActive }) =>
                  `block py-2 px-4 rounded ${
                    isActive ? "underline decoration-blue-500" : ""
                  }`
                }
              >
                Home
              </NavLink>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <NavLink
                to="/about"
                className={({ isActive }) =>
                  `block py-2 px-4 rounded ${
                    isActive ? "underline decoration-blue-500" : ""
                  }`
                }
              >
                About
              </NavLink>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <NavLink
                to="/contact"
                className={({ isActive }) =>
                  `block py-2 px-4 rounded ${
                    isActive ? "underline decoration-blue-500" : ""
                  }`
                }
              >
                Contact
              </NavLink>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <NavLink
                to="/upload"
                className={({ isActive }) =>
                  `block py-2 px-4 rounded ${
                    isActive ? "underline decoration-blue-500" : ""
                  }`
                }
              >
                Upload
              </NavLink>
            </NavigationMenuItem>
            {isAdminAuthenticated && (
              <NavigationMenuItem>
                <NavLink
                  to="/dashboard"
                  className={({ isActive }) =>
                    `block py-2 px-4 rounded ${
                      isActive ? "underline decoration-blue-500" : ""
                    }`
                  }
                >
                  Dashboard
                </NavLink>
              </NavigationMenuItem>
            )}
            <NavigationMenuItem>
              <button onClick={handleLogout} className="py-2 px-4 rounded">
                Logout
              </button>
            </NavigationMenuItem>
          </NavigationMenuList>
        </NavigationMenu>

        {/* Mobile Menu Toggle Button */}
        <div className="md:hidden">
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            className="focus:outline-none"
          >
            {menuOpen ? <X size={28} /> : <Menu size={28} />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation */}
      {menuOpen && (
        <div className="flex flex-col space-y-3 mt-4 px-6 md:hidden">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `block py-2 px-4 rounded ${
                isActive ? "underline decoration-blue-500" : ""
              }`
            }
            onClick={() => setMenuOpen(false)}
          >
            Home
          </NavLink>
          <NavLink
            to="/about"
            className={({ isActive }) =>
              `block py-2 px-4 rounded ${
                isActive ? "underline decoration-blue-500" : ""
              }`
            }
            onClick={() => setMenuOpen(false)}
          >
            About
          </NavLink>
          <NavLink
            to="/contact"
            className={({ isActive }) =>
              `block py-2 px-4 rounded ${
                isActive ? "underline decoration-blue-500" : ""
              }`
            }
            onClick={() => setMenuOpen(false)}
          >
            Contact
          </NavLink>

          <NavLink
            to="/upload"
            className={({ isActive }) =>
              `block py-2 px-4 rounded ${
                isActive ? "underline decoration-blue-500" : ""
              }`
            }
            onClick={() => setMenuOpen(false)}
          >
            Upload
          </NavLink>
          {isAdminAuthenticated && (
            <NavLink
              to="/dashboard"
              className={({ isActive }) =>
                `block py-2 px-4 rounded ${
                  isActive ? "underline decoration-blue-500" : ""
                }`
              }
              onClick={() => setMenuOpen(false)}
            >
              Dashboard
            </NavLink>
          )}
          <button
            onClick={() => {
              handleLogout();
              setMenuOpen(false);
            }}
            className="block py-2 px-4"
          >
            Logout
          </button>
        </div>
      )}
    </nav>
  ) : null;
}
