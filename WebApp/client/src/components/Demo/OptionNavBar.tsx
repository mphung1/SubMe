import { Outlet } from "react-router-dom";
import { Nav, NavLink, NavMenu } from "./MenuBarElements";

interface LinkItemProps {
  key: number;
  name: string;
  path: string;
}
const LinkItems: Array<LinkItemProps> = [
  { key: 1, name: "Search", path: "" },
  { key: 2, name: "Upload", path: "" },
  { key: 3, name: "By URL", path: "" },
];

function OptionNavBar() {
  return (
    <>
      <Nav>
        <NavMenu>
          {LinkItems.map((link) => (
            <NavLink key={link.key} to={link.path}>
              {link.name}
            </NavLink>
          ))}
        </NavMenu>
      </Nav>
      <hr />
      <Outlet />
    </>
  );
}

export default OptionNavBar;
